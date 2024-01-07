from mmcv.ops import RoIAlign
from mmdet.models import CrossEntropyLoss, DiceLoss, FPN, SingleRoIExtractor
from mmengine.config import read_base

from seg.models.detectors import CLIP2SAM
from seg.models.backbones import OpenCLIPBackbone
from seg.models.necks import MultiLayerTransformerNeck, SAMPromptEncoder
from seg.models.heads import OVSAMHead
from seg.models.data_preprocessor import OVSAMDataPreprocessor
from seg.models.utils import NO_OBJ

with read_base():
    from .._base_.default_runtime import *
    from .._base_.datasets.coco_ov_instance_lsj import *
    from .._base_.schedules.schedule_12e import *

image_size = (1024, 1024)
data_preprocessor = dict(
    type=OVSAMDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1024,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=False,
    seg_pad_value=NO_OBJ,
    batch_augments=None,
    use_point_det=True,
    num_proposals=40,
)

model = dict(
    type=CLIP2SAM,
    data_preprocessor=data_preprocessor,
    with_box=True,
    with_points=True,
    backbone=dict(
        type=OpenCLIPBackbone,
        model_name='RN50x16',
        fix=True,
        init_cfg=dict(
            type='clip_pretrain',
            checkpoint='openai'
        )
    ),
    neck=dict(
        type=MultiLayerTransformerNeck,
        input_size=(1024, 1024),
        in_channels=[384, 768, 1536, 3072],
        strides=[4, 8, 16, 32],
        layer_ids=(0, 1, 2, 3),
        embed_channels=1280,
        out_channels=256,
        fix=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./models/sam2clip_vith_rn50x16.pth',
            prefix='neck_student',
        )
    ),
    fpn_neck=dict(
        type=FPN,
        in_channels=[384, 768, 1536, 3072],
        out_channels=256,
        num_outs=4,
    ),
    prompt_encoder=dict(
        type=SAMPromptEncoder,
        model_name='vit_h',
        fix=True,
        init_cfg=dict(
            type='sam_pretrain',
            checkpoint='vit_h'
        )
    ),
    mask_decoder=dict(
        type=OVSAMHead,
        model_name='vit_h',
        with_label_token=True,
        ov_classifier_name='RN50x16_CocoOVDataset',
        roi_extractor=dict(
            type=SingleRoIExtractor,
            roi_layer=dict(type=RoIAlign, output_size=12, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        fix=False,
        init_cfg=dict(
            type='sam_pretrain',
            checkpoint='vit_h'
        ),
        loss_cls=dict(
            type=CrossEntropyLoss,
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean'
        ),
        loss_mask=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0
        ),
        loss_dice=dict(
            type=DiceLoss,
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0
        )
    )
)

val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = None
test_evaluator = None
test_cfg = None
