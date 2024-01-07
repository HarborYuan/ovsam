from typing import Union, Tuple, Dict, List

import torch
from mmdet.models.detectors.base import ForwardResults
from mmengine.model import BaseModel
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList, OptSampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class CLIP2SAM(BaseModel):
    MASK_THRESHOLD = 0.5

    def __init__(
            self,
            backbone: ConfigType,
            neck: ConfigType,
            prompt_encoder: ConfigType,
            mask_decoder: ConfigType,
            fpn_neck: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            with_box: bool = True,
            with_points: bool = True,
            init_cfg: OptMultiConfig = None,
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

        self.with_box = with_box
        self.with_points = with_points

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

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> Union[Dict, List]:
        raise NotImplementedError

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[Dict, List]:
        backbone_feats = self.backbone(batch_inputs)
        feats = self.neck(backbone_feats)

        if self.fpn_neck is not None:
            fpn_feats = self.fpn_neck(backbone_feats)
        else:
            fpn_feats = None

        spase_embed_list = []
        dense_embed_list = []
        batch_ind_list = []
        img_size = batch_data_samples[0].batch_input_shape
        for idx, data_sample in enumerate(batch_data_samples):
            if self.with_box and self.with_points:
                is_box = data_sample.gt_instances.bp.eq(0)
                is_point = data_sample.gt_instances.bp.eq(1)
                assert is_box.any()
                sparse_embed, dense_embed = self.pe(
                    data_sample.gt_instances[is_box],
                    image_size=img_size,
                    with_bboxes=True,
                    with_points=False,
                )
                sparse_embed, dense_embed = [sparse_embed], [dense_embed]
                if is_point.any():
                    _sparse_embed, _dense_embed = self.pe(
                        data_sample.gt_instances[is_point],
                        image_size=img_size,
                        with_bboxes=False,
                        with_points=True,
                    )
                    sparse_embed.append(_sparse_embed)
                    dense_embed.append(_dense_embed)
                sparse_embed, dense_embed = torch.cat(sparse_embed), torch.cat(dense_embed)
            else:
                sparse_embed, dense_embed = self.pe(
                    data_sample.gt_instances,
                    image_size=img_size,
                    with_bboxes=self.with_box,
                    with_points=self.with_points,
                )
            assert len(sparse_embed) == len(dense_embed)
            assert len(sparse_embed) == len(data_sample.gt_instances)
            spase_embed_list.append(sparse_embed)
            dense_embed_list.append(dense_embed)
            batch_ind_list.append(len(sparse_embed))

        sparse_embed = torch.cat(spase_embed_list)
        dense_embed = torch.cat(dense_embed_list)

        if fpn_feats is not None:
            losses = self.mask_decoder.forward_train(
                image_embeddings=feats,
                image_pe=self.pe.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                batch_ind_list=batch_ind_list,
                data_samples=batch_data_samples,
                fpn_feats=fpn_feats,
            )
        else:
            losses = self.mask_decoder.forward_train(
                image_embeddings=feats,
                image_pe=self.pe.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embed,
                dense_prompt_embeddings=dense_embed,
                batch_ind_list=batch_ind_list,
                data_samples=batch_data_samples,
            )
        return losses
