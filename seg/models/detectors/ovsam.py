from typing import Union, Tuple, Dict, List

import torch
import torch.nn.functional as F
from mmcv.ops import RoIAlign
from mmdet.models.detectors.base import ForwardResults
from mmdet.structures.bbox import bbox2roi
from mmengine.model import BaseModel
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList, OptSampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from seg.models.utils import mask_pool


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
class OVSAM(BaseModel):
    MASK_THRESHOLD = 0.5

    def __init__(
            self,
            backbone: ConfigType,
            neck: ConfigType,
            prompt_encoder: ConfigType,
            mask_decoder: ConfigType,
            data_preprocessor: OptConfigType = None,
            fpn_neck: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
            use_clip_feat: bool = False,
            use_head_feat: bool = False,
            use_gt_prompt: bool = False,
            use_point: bool = False,
            enable_backbone:bool = False,
            alpha: float = .1,
            beta: float = .9,
            num_classes=0,
            base_classes=None,
            novel_classes=None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)

        self.pe = MODELS.build(prompt_encoder)

        self.mask_decoder = MODELS.build(mask_decoder)

        if fpn_neck is not None:
            self.fpn_neck = MODELS.build(fpn_neck)
        else:
            self.fpn_neck = None

        self.use_clip_feat = use_clip_feat
        self.use_head_feat = use_head_feat
        self.use_gt_prompt = use_gt_prompt
        self.use_point = use_point

        self.enable_backbone = enable_backbone

        self.alpha = alpha
        self.beta = beta

        if self.backbone.feat_size > 0:
            self.roi = RoIAlign(
                output_size=(self.backbone.feat_size, self.backbone.feat_size),
                spatial_scale=1 / 32.,
            )
        else:
            self.roi = None

        self.num_classes = num_classes
        self.base_novel_indicator = None
        if base_classes is not None:
            assert novel_classes is not None
            self.base_novel_indicator = torch.zeros((self.num_classes,), dtype=torch.long)
            if len(base_classes) > 0:
                self.base_novel_indicator[torch.tensor(base_classes)] = 1
            if len(novel_classes) > 0:
                self.base_novel_indicator[torch.tensor(novel_classes)] = 2

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
        batch_data_samples[0].gt_instances = batch_data_samples[0].gt_instances[:10]
        results = self.predict(batch_inputs, batch_data_samples)
        return results[0].pred_instances.masks.sum() + results[0].pred_instances.labels.sum()

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> Union[Dict, List]:
        backbone_feat = self.backbone(batch_inputs)
        batch_feats = self.neck(backbone_feat)
        if self.fpn_neck is not None:
            fpn_feats = self.fpn_neck(backbone_feat)
        else:
            fpn_feats = None

        for feat, data_sample in zip(batch_feats, batch_data_samples):
            if self.use_gt_prompt:
                prompt_instances = data_sample.gt_instances
                data_sample.pred_instances = InstanceData()
            else:
                prompt_instances = data_sample.pred_instances
            if len(prompt_instances) == 0:
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
            kwargs = {}
            if self.enable_backbone:
                kwargs['backbone_feats'] = backbone_feat
                kwargs['backbone'] = self.backbone
            if fpn_feats is not None:
                assert len(batch_data_samples) == 1
                low_res_masks, iou_predictions, cls_pred = self.mask_decoder(
                    image_embeddings=feat.unsqueeze(0),
                    image_pe=self.pe.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embed,
                    dense_prompt_embeddings=dense_embed,
                    multi_mask_output=False,
                    fpn_feats=[itm[0:1] for itm in fpn_feats],
                    data_samples=data_sample,
                    **kwargs
                )
            else:
                low_res_masks, iou_predictions, cls_pred = self.mask_decoder(
                    image_embeddings=feat.unsqueeze(0),
                    image_pe=self.pe.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embed,
                    dense_prompt_embeddings=dense_embed,
                    multi_mask_output=False,
                    **kwargs
                )

            cls_pred = self.open_voc_inference(backbone_feat, cls_pred, low_res_masks, data_samples=data_sample)

            masks = postprocess_masks(
                masks=low_res_masks,
                pad_size=data_sample.batch_input_shape,
                input_size=data_sample.img_shape,
                original_size=data_sample.ori_shape,
            )
            masks = masks.sigmoid() > self.MASK_THRESHOLD

            results = InstanceData()
            results.masks = masks[:, 0]
            if not self.use_gt_prompt:
                scale_factor = data_sample.pred_instances.bboxes.new_tensor(data_sample.scale_factor).repeat(2)
                results.scores = data_sample.pred_instances.scores
                results.bboxes = data_sample.pred_instances.bboxes / scale_factor

            if cls_pred is not None:
                results.labels = cls_pred[:, 0].softmax(-1)[:, :-1].argmax(-1)
            else:
                results.labels = data_sample.pred_instances.labels
            data_sample.pred_instances = results
        return batch_data_samples

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[Dict, List]:
        raise NotImplementedError

    def open_voc_inference(self, feats, mask_cls_results, mask_pred_results, data_samples=None):
        if self.use_head_feat:
            query_logit = mask_cls_results
        else:
            query_logit = None

        if self.use_clip_feat:
            clip_feat = self.backbone.get_clip_feature(feats[-1])
            clip_feat_mask = F.interpolate(
                mask_pred_results,
                size=clip_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            if self.roi is not None:
                if data_samples is not None:
                    bboxes = data_samples.gt_instances.bboxes
                    instance_feat = self.roi(clip_feat, bbox2roi([bboxes]))
                else:
                    raise NotImplementedError
            else:
                instance_feat = mask_pool(clip_feat, clip_feat_mask)
            instance_feat = self.backbone.forward_feat(instance_feat)
            if len(instance_feat.shape) == 2:
                instance_feat = instance_feat[:, None]
            clip_logit = self.mask_decoder.forward_logit(instance_feat)
        else:
            clip_logit = None

        if query_logit is None and clip_logit is None:
            return None

        if query_logit is None:
            return clip_logit

        if clip_logit is None:
            return query_logit

        clip_logit = clip_logit.softmax(-1)
        query_logit = query_logit.softmax(-1)

        classes_inference = clip_logit.size(-1) - 1
        if self.base_novel_indicator is not None:
            assert classes_inference == self.num_classes
            overlapping_mask = torch.zeros(query_logit.size(-1), dtype=torch.float32, device=clip_logit.device)
            overlapping_mask[-1] = 1.
            overlapping_mask[..., :-1][self.base_novel_indicator == 1] = 1
        elif hasattr(self, "OVERLAPPING"):
            overlapping_mask = torch.tensor(self.OVERLAPPING, dtype=torch.float32, device=clip_logit.device)
            overlapping_mask = torch.cat([
                overlapping_mask, torch.ones((1,), dtype=torch.float32, device=clip_logit.device)]
            )
        else:
            overlapping_mask = torch.zeros(query_logit.size(-1), dtype=torch.float32, device=clip_logit.device)
            overlapping_mask[-1] = 1.

        valid_masking = ((clip_feat_mask > 0).to(
            dtype=torch.float32).flatten(-2).sum(-1) > 0).to(torch.float32)[..., None]
        alpha = torch.ones_like(clip_logit) * self.alpha * valid_masking
        beta = torch.ones_like(clip_logit) * self.beta * valid_masking

        cls_logits_seen = (
                (query_logit ** (1 - alpha) * clip_logit ** alpha).log() * overlapping_mask
        )
        cls_logits_unseen = (
                (query_logit ** (1 - beta) * clip_logit ** beta).log() * (1 - overlapping_mask)
        )
        cls_results = cls_logits_seen + cls_logits_unseen
        if self.base_novel_indicator is not None:
            cls_results[..., :-1][..., self.base_novel_indicator == 0] = -1000.
        return cls_results
