from mmdet.models import BatchFixedSizePad, DetDataPreprocessor, MSELoss
from mmengine.config import read_base

from projects.rwkvsam.models import BackboneDistillation, SAMBackbone, LastLayerNeck, LastLayerProjNeck, VITAMINBackbone

with read_base():
    from .._base_.default_runtime import *
    from .._base_.datasets.sam.sam_distill import *
    from .._base_.schedules.schedule_24e_distillation import *

batch_augments = [
    dict(
        type=BatchFixedSizePad,
        size=image_size,
        img_pad_value=0,
    )
]
data_preprocessor = dict(
    type=DetDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1,
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
        type=VITAMINBackbone,
        img_size=(224, 224),
        model_variant='small',
        attn_type='rwkv',
        attn_cfg=dict(
            mlp_ratio=2,
        ),
        with_pos_embd=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs/ckpt/ssmseg_pretrain_vitamin_rwkv_mlp2_pretrain_16xbs64_best_accuracy_top1_epoch_294.pth',
            prefix='backbone.',
        )
    ),
    neck_teacher=dict(type=LastLayerNeck),
    neck_student=dict(
        type=LastLayerProjNeck,
        in_channels=384,
        out_channels=256,
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
