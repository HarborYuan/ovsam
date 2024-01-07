from typing import Tuple

from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet.registry import MODELS

from ext.sam.common import LayerNorm2d
from seg.models.utils.load_checkpoint import load_checkpoint_with_prefix


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

        if init_cfg is not None and init_cfg['type'] == 'Pretrained':
            checkpoint_path = init_cfg['checkpoint']
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=True)
            self._is_init = True

    def init_weights(self):
        pass

    def forward(self, inputs: Tuple[Tensor]) -> Tensor:
        return self.out_proj(inputs[-1])
