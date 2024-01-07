from mmdet.models import BatchFixedSizePad, DetDataPreprocessor, MSELoss
from mmengine.config import read_base

from seg.models.detectors import BackboneDistillation
from seg.models.backbones import OpenCLIPBackbone, SAMBackbone
from seg.models.necks import LastLayerNeck
from seg.models.necks.transformer_neck import MultiLayerTransformerNeck
from seg.models.utils import NO_OBJ

with read_base():
    from .._base_.default_runtime import *
    from .._base_.datasets.sam_img import *
    from .._base_.schedules.schedule_distillation import *

image_size = (1024, 1024)
batch_augments = [
    dict(
        type=BatchFixedSizePad,
        size=image_size,
        img_pad_value=0,
        pad_mask=False,
        mask_pad_value=0,
        pad_seg=False,
        seg_pad_value=255
    )
]
data_preprocessor = dict(
    type=DetDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1,
    pad_mask=False,
    mask_pad_value=0,
    pad_seg=False,
    seg_pad_value=NO_OBJ,
    batch_augments=batch_augments
)

model = dict(
    type=BackboneDistillation,
    use_cache=True,
    data_preprocessor=data_preprocessor,
    backbone_teacher=dict(
        type=SAMBackbone,
        model_name='vit_h',
        fix=True,
        init_cfg=dict(
            type='sam_pretrain',
            checkpoint='vit_h'
        )
    ),
    backbone_student=dict(
        type=OpenCLIPBackbone,
        model_name='RN50',
        fix=True,
        init_cfg=dict(
            type='clip_pretrain',
            checkpoint='openai'
        )
    ),
    neck_teacher=dict(type=LastLayerNeck),
    neck_student=dict(
        type=MultiLayerTransformerNeck,
        input_size=(1024, 1024),
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        layer_ids=(0, 1, 2, 3),
        embed_channels=1280,
        out_channels=256,
        embedding_path='sam_vit_h'
    ),
    loss_distill=dict(
        type=MSELoss,
        reduction='mean',
        loss_weight=1.
    )
)

val_dataloader = None
val_evaluator = None
val_cfg = None
