from typing import Union, Tuple, Dict, List

import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer
from mmdet.models.detectors.base import ForwardResults
from mmengine import print_log
from mmengine.model import BaseModel
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList, OptSampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class BackboneDistillation(BaseModel):

    def __init__(
            self,
            backbone_teacher: ConfigType,
            backbone_student: ConfigType,
            neck_teacher: ConfigType,
            neck_student: ConfigType,
            loss_distill: ConfigType,
            add_adapter: bool = False,
            use_cache: bool = False,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.use_cache = use_cache
        if not self.use_cache:
            self.backbone_teacher = MODELS.build(backbone_teacher)
            self.neck_teacher = MODELS.build(neck_teacher)
        else:
            self.cache_suffix = f'_{backbone_teacher.model_name}_cache.pth'

        self.backbone_student = MODELS.build(backbone_student)
        self.neck_student = MODELS.build(neck_student)

        self.loss_distill = MODELS.build(loss_distill)

        self.add_adapter = add_adapter
        if self.add_adapter:
            STRIDE = 16
            self.patch_embeds = nn.ModuleList()
            for stride in [4, 8, 16, 32]:
                if stride > 16:
                    patch_embed = build_conv_layer(
                        dict(type=nn.ConvTranspose2d),
                        in_channels=256,
                        out_channels=256,
                        kernel_size=stride // STRIDE,
                        stride=stride // STRIDE,
                        padding=0,
                        dilation=1,
                        bias=0
                    )
                else:
                    patch_embed = build_conv_layer(
                        dict(type=nn.Conv2d),
                        in_channels=256,
                        out_channels=256,
                        kernel_size=STRIDE // stride,
                        stride=STRIDE // stride,
                        padding=0,
                        dilation=1,
                        bias=0
                    )
                self.patch_embeds.append(patch_embed)

        print_log(
            f"{self.__class__.__name__} -> teacher: {backbone_teacher.model_name}; "
            f"student: {self.backbone_student.__class__.__name__}",
            logger='current',
        )

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

    def _forward(self, *args, **kwargs) -> Tuple[Tensor]:
        raise NotImplementedError

    def extract_feat(self, batch_inputs: Tensor, batch_data_samples) -> Tuple[Tensor, Tensor]:
        if self.use_cache:
            feat_list = []
            for data_samples in batch_data_samples:
                feat_list.append(data_samples.gt_feats)
            feat_teacher = torch.stack(feat_list)
        else:
            feat_teacher = self.neck_teacher(self.backbone_teacher(batch_inputs))
        feat_student = self.neck_student(self.backbone_student(batch_inputs))
        if self.add_adapter:
            feat_list = []
            for idx in range(4):
                feat_list.append(self.patch_embeds[idx](feat_student[idx]))
            feat_student = torch.stack(feat_list, dim=0).mean(dim=0)
        return feat_teacher, feat_student

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[Dict, List]:
        feat_teacher, feat_student = self.extract_feat(batch_inputs, batch_data_samples)
        return {
            "loss_distillation": self.loss_distill(feat_teacher, feat_student)
        }
