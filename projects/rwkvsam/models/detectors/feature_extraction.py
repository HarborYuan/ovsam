from typing import Optional, Union, Dict

import torch
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType
from mmengine.model import BaseModel
from mmengine.structures import InstanceData


@MODELS.register_module()
class FeatExtraction(BaseModel):

    def __init__(
            self,
            data_preprocessor,
            backbone: OptConfigType = None,
            init_cfg=None,
    ):
        super(FeatExtraction, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        feats = self.backbone(inputs)
        return feats


@MODELS.register_module()
class MaskExtraction(BaseModel):
    def __init__(
            self,
            data_preprocessor,
            backbone: OptConfigType,
            neck: OptConfigType,
            prompt_encoder: OptConfigType,
            mask_decoder: OptConfigType,
            init_cfg=None,
    ):
        super(MaskExtraction, self).__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.pe = MODELS.build(prompt_encoder)
        self.mask_decoder = MODELS.build(mask_decoder)

        self.add_extra_hr_feat = True
        self.num_ins = 1

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        backbone_feat = self.backbone(inputs)
        batch_feats = self.neck(backbone_feat)
        prompt_instances = InstanceData(
            bboxes=torch.tensor([[0, 0, 1, 1]], dtype=torch.float32, device=batch_feats.device)
        )
        sparse_embed, dense_embed = self.pe(
            prompt_instances,
            image_size=data_samples[0].batch_input_shape,
            with_bboxes=True,
        )

        kwargs = dict()
        if self.add_extra_hr_feat:
            kwargs['hr_feat'] = backbone_feat[0]
            kwargs['mr_feat'] = backbone_feat[1]

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=batch_feats,
            image_pe=self.pe.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embed,
            dense_prompt_embeddings=dense_embed,
            multi_mask_output=False,
            **kwargs
        )
        return low_res_masks
