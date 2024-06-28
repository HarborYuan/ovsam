from typing import Tuple

from mmengine.model import BaseModule
from torch import Tensor, nn
import torch.nn.functional as F

from mmdet.registry import MODELS

from ext.sam.common import LayerNorm2d
from projects.rwkvsam.utils import load_checkpoint_with_prefix


@MODELS.register_module()
class LastLayerNeck(BaseModule):
    r"""Last Layer Neck

    Return the last layer feature of the backbone.
    """

    def __init__(self) -> None:
        super().__init__(init_cfg=None)

    def forward(self, inputs: Tuple[Tensor]) -> Tensor:
        return inputs[-1]


@MODELS.register_module()
class LastLayerProjNeck(BaseModule):

    def __init__(
            self,
            in_channels,
            out_channels,
            scale_factor: int = 1,
            init_cfg=None
    ) -> None:
        super().__init__(init_cfg=None)
        self.out_proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
        )

        self.scale_factor = scale_factor

        if init_cfg is not None and init_cfg['type'] == 'Pretrained':
            checkpoint_path = init_cfg['checkpoint']
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=True)
            self._is_init = True

    def init_weights(self):
        pass

    def forward(self, inputs: Tuple[Tensor]) -> Tensor:
        feat = self.out_proj(inputs[-1])
        if self.scale_factor != 1:
            feat = F.interpolate(
                feat,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=False
            )
        return feat


@MODELS.register_module()
class LastTwoLayerProjNeck(BaseModule):

    def __init__(
            self,
            in_channels: Tuple[int, int],
            out_channels: int,
            init_cfg=None
    ) -> None:
        super().__init__(init_cfg=None)
        self.out_proj1 = nn.Sequential(
            nn.Conv2d(
                in_channels[0],
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
        )
        self.out_proj2 = nn.Sequential(
            nn.Conv2d(
                in_channels[1],
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
        )

        if init_cfg is not None and init_cfg['type'] == 'Pretrained':
            checkpoint_path = init_cfg['checkpoint']
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=True)
            self._is_init = True

    def init_weights(self):
        pass

    def forward(self, inputs: Tuple[Tensor]) -> Tensor:
        feat = inputs[-1]
        feat = self.out_proj2(feat)
        feat = F.interpolate(
            feat,
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )
        feat = feat + self.out_proj1(inputs[-2])
        return feat
