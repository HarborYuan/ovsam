from typing import Union, Tuple, Dict, List

import torch
import torch.nn.functional as F
from mmdet.models.detectors.base import ForwardResults
from mmengine.model import BaseModel
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList, OptSampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


def postprocess_masks(
        masks: torch.Tensor,
        pad_size: Tuple[int, int],
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
) -> torch.Tensor:
    masks = F.interpolate(
        masks,
        pad_size,
        mode="bilinear",
        align_corners=False,
    )
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks


@MODELS.register_module()
class SAMModel(BaseModel):
    MASK_THRESHOLD = 0.5

    def __init__(
            self,
            backbone: ConfigType,
            neck: ConfigType,
            prompt_encoder: ConfigType,
            mask_decoder: ConfigType,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
            # eval settings
            use_point: bool = False,
            use_gt_prompt: bool = False,
            use_multi: bool = False,
            # extra info
            add_extra_hr_feat: bool = False,
            add_backbone_feat: bool = False,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.pe = MODELS.build(prompt_encoder)
        self.mask_decoder = MODELS.build(mask_decoder)

        # Eval settings
        self.use_point = use_point
        self.use_gt_prompt = use_gt_prompt
        self.use_multi = use_multi

        # Extra info
        self.add_extra_hr_feat = add_extra_hr_feat
        self.add_backbone_feat = add_backbone_feat

        self.tmp_direct_output = False

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.neck(self.backbone(batch_inputs))
        return x

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList = None) -> Tuple[Tensor]:
        raise NotImplementedError

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[Dict, List]:
        backbone_feat = self.backbone(batch_inputs)
        batch_feats = self.neck(backbone_feat)

        spase_embed_list = []
        dense_embed_list = []
        batch_ind_list = []
        img_size = batch_data_samples[0].batch_input_shape
        for idx, data_sample in enumerate(batch_data_samples):
            is_box = data_sample.gt_instances.bp.eq(0)
            is_point = data_sample.gt_instances.bp.eq(1)
            is_mask = data_sample.gt_instances.bp.eq(2)
            sparse_embed, dense_embed = [], []
            if is_box.any():
                _sparse_embed, _dense_embed = self.pe(
                    data_sample.gt_instances[is_box],
                    image_size=img_size,
                    with_bboxes=True,
                    with_points=False,
                    with_masks=False,
                )
                sparse_embed.append(_sparse_embed)
                dense_embed.append(_dense_embed)
            if is_point.any():
                _sparse_embed, _dense_embed = self.pe(
                    data_sample.gt_instances[is_point],
                    image_size=img_size,
                    with_bboxes=False,
                    with_points=True,
                    with_masks=False,
                )
                sparse_embed.append(_sparse_embed)
                dense_embed.append(_dense_embed)
            if is_mask.any():
                raise NotImplementedError
            sparse_embed, dense_embed = torch.cat(sparse_embed), torch.cat(dense_embed)
            assert len(sparse_embed) > 0
            assert len(sparse_embed) == len(dense_embed)
            assert len(sparse_embed) == len(data_sample.gt_instances)
            spase_embed_list.append(sparse_embed)
            dense_embed_list.append(dense_embed)
            batch_ind_list.append(len(sparse_embed))

        sparse_embed = torch.cat(spase_embed_list)
        dense_embed = torch.cat(dense_embed_list)

        kwargs = dict()
        if self.add_extra_hr_feat:
            kwargs['hr_feat'] = backbone_feat[0]
            kwargs['mr_feat'] = backbone_feat[1]

        if self.add_backbone_feat:
            kwargs['backbone_feat'] = backbone_feat

        losses = self.mask_decoder.forward_train(
            image_embeddings=batch_feats,
            image_pe=self.pe.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embed,
            dense_prompt_embeddings=dense_embed,
            batch_ind_list=batch_ind_list,
            data_samples=batch_data_samples,
            **kwargs,
        )
        return losses

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> Union[Dict, List]:
        backbone_feat = self.backbone(batch_inputs)
        batch_feats = self.neck(backbone_feat)

        for feat, data_sample in zip(batch_feats, batch_data_samples):
            if self.use_gt_prompt:
                prompt_instances = data_sample.gt_instances
                data_sample.pred_instances = InstanceData()
            else:
                prompt_instances = data_sample.pred_instances

            # if not instances, just skip the following process
            if len(prompt_instances) == 0:
                data_sample.pred_instances.masks = torch.zeros((0, *data_sample.ori_shape), dtype=torch.bool)
                continue

            if self.use_point:
                sparse_embed, dense_embed = self.pe(
                    prompt_instances,
                    image_size=data_sample.batch_input_shape,
                    with_points=True,
                )
            else:
                sparse_embed, dense_embed = self.pe(
                    prompt_instances,
                    image_size=data_sample.batch_input_shape,
                    with_bboxes=True,
                )

            kwargs = dict()
            if self.add_extra_hr_feat:
                kwargs['hr_feat'] = backbone_feat[0]
                kwargs['mr_feat'] = backbone_feat[1]

            if self.add_backbone_feat:
                kwargs['backbone_feat'] = backbone_feat

            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=feat.unsqueeze(0),
                image_pe=self.pe.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                multi_mask_output=self.use_multi,
                **kwargs
            )

            if self.use_multi:
                sorted_ids = torch.argsort(iou_predictions, dim=-1, descending=True)
                _ = torch.take_along_dim(iou_predictions, sorted_ids, dim=-1)
                low_res_masks = torch.take_along_dim(
                    low_res_masks, sorted_ids[..., None, None], dim=-3
                )
                low_res_masks = low_res_masks[:, 0:1]

            masks = postprocess_masks(
                masks=low_res_masks,
                pad_size=data_sample.batch_input_shape,
                input_size=data_sample.img_shape,
                original_size=data_sample.ori_shape,
            )
            masks = masks.sigmoid() > self.MASK_THRESHOLD

            if self.tmp_direct_output:
                # FIXME: Delete it
                img_filename = data_sample.img_id
                import os
                import numpy as np
                import mmcv
                img_dir = os.path.join('./vissam', img_filename)
                mask = masks[0][0].cpu().numpy().astype(np.uint8) * 255
                mmcv.imwrite(mask, img_dir)

            results = InstanceData()
            results.masks = masks[:, 0]
            if self.use_gt_prompt:
                scale_factor = data_sample.gt_instances.bboxes.new_tensor(data_sample.scale_factor).repeat(2)
                results.bboxes = data_sample.gt_instances.bboxes / scale_factor
                results.labels = data_sample.gt_instances.labels
            else:
                scale_factor = data_sample.pred_instances.bboxes.new_tensor(data_sample.scale_factor).repeat(2)
                results.scores = data_sample.pred_instances.scores
                results.bboxes = data_sample.pred_instances.bboxes / scale_factor
                results.labels = data_sample.pred_instances.labels
            data_sample.pred_instances = results
        return batch_data_samples
