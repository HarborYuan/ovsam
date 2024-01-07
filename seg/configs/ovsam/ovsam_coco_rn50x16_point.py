from mmcv.ops import RoIAlign
from mmdet.models import FPN, SingleRoIExtractor
from mmengine.config import read_base

from seg.models.data_preprocessor import OVSAMDataPreprocessor
from seg.models.backbones import OpenCLIPBackbone
from seg.models.detectors import OVSAM
from seg.models.heads import OVSAMHead
from seg.models.necks import SAMPromptEncoder, MultiLayerTransformerNeck

with read_base():
    from .._base_.default_runtime import *
    from .._base_.datasets.coco_ov_instance_lsj import *
    from .._base_.schedules.schedule_12e import *

image_size = (1024, 1024)
_data_preprocessor = dict(
    type=OVSAMDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=image_size[0],
    pad_mask=False,
    mask_pad_value=0,
    pad_seg=False,
    seg_pad_value=255,
    batch_augments=None,
    use_center_point=True
)
model = dict(
    type=OVSAM,
    data_preprocessor=_data_preprocessor,
    use_gt_prompt=True,
    use_clip_feat=True,
    use_head_feat=True,
    use_point=True,
    num_classes=80,
    base_classes=COCO4817_BASE_IDS,
    novel_classes=COCO4817_NOVEL_IDS,
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
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./models/clip2sam_coco_rn50x16.pth',
            prefix='fpn_neck',
        ),
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
        gen_box=True,
        model_name='vit_h',
        with_label_token=True,
        fix=False,
        ov_classifier_name='RN50x16_CocoOVDataset',
        roi_extractor=dict(
            type=SingleRoIExtractor,
            roi_layer=dict(type=RoIAlign, output_size=12, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./models/clip2sam_coco_rn50x16.pth',
            prefix='mask_decoder',
        )
    )
)
