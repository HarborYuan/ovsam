from typing import Union, Tuple, Dict, List

import mmengine
import torch
from mmdet.models.detectors.base import ForwardResults
from mmengine import print_log
from mmengine.model import BaseModel
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList, OptSampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class BackboneDump(BaseModel):

    def __init__(
            self,
            backbone: ConfigType,
            neck: ConfigType,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)

        self.register_parameter('dummy', torch.nn.Parameter(torch.zeros(1)))

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

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor, Tensor]:
        return self.neck(self.backbone(batch_inputs))

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> Union[Dict, List]:
        feat = self.extract_feat(batch_inputs)

        assert len(batch_data_samples) == 1
        img_path = batch_data_samples[0].metainfo['img_path']
        img_path = img_path.replace('.jpg', f'_{self.backbone.model_name}_cache.pth')
        if not mmengine.exists(img_path):
            feat = feat.to(device='cpu')[0]
            torch.save(feat.to(device='cpu'), img_path)
        else:
            print_log(f'{img_path} already exists')
        return {}

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[Dict, List]:
        raise NotImplementedError
