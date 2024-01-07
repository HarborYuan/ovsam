from mmdet.models import BatchFixedSizePad, DetDataPreprocessor
from mmengine.config import read_base

from seg.models.backbones import SAMBackbone
from seg.models.detectors import BackboneDump
from seg.models.necks import LastLayerNeck
from seg.models.utils import NO_OBJ

with read_base():
    from .._base_.default_runtime import *
    from .._base_.datasets.sam import *
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
        seg_pad_value=NO_OBJ
    )
]
data_preprocessor = dict(
    type=DetDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1024,
    pad_mask=False,
    mask_pad_value=0,
    pad_seg=False,
    seg_pad_value=NO_OBJ,
    batch_augments=batch_augments
)

model = dict(
    type=BackboneDump,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type=SAMBackbone,
        model_name='vit_h',
        fix=True,
        init_cfg=dict(
            type='sam_pretrain',
            checkpoint='vit_h'
        )
    ),
    neck=dict(
        type=LastLayerNeck
    )
)

val_dataloader = None
val_evaluator = None
val_cfg = None
