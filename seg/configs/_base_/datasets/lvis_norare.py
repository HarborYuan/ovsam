# dataset settings
from mmcv import LoadImageFromFile, RandomResize
from mmdet.datasets import LVISV1Dataset, AspectRatioBatchSampler
from mmdet.datasets.transforms import LoadAnnotations, Resize, RandomFlip, PackDetInputs, RandomCrop
from mmengine.dataset import DefaultSampler

from seg.datasets.pipeliens.loading import FilterAnnotationsHB
from seg.evaluation.ins_cls_iou_metric import InsClsIoUMetric

from ext.class_names.lvis_ids import LVIS_BASE_IDS, LVIS_RARE_IDS

data_root = 'data/lvis/'
backend_args = None
dataset_type = LVISV1Dataset

image_size = (1024, 1024)

train_pipeline = [
    dict(
        type=LoadImageFromFile,
        to_float32=True,
        backend_args=backend_args),
    dict(
        type=LoadAnnotations,
        with_bbox=True,
        with_mask=True,
        backend_args=backend_args),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=RandomResize,
        resize_type=Resize,
        scale=image_size,
        ratio_range=(.1, 2.),
        keep_ratio=True,
    ),
    dict(
        type=RandomCrop,
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(
        type=FilterAnnotationsHB,
        by_box=False,
        by_mask=True,
        min_gt_mask_area=32,
    ),
    dict(type=PackDetInputs)
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/lvis_v1_train_norare.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args)
)

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1024, 1024), keep_ratio=True),
    dict(type=LoadAnnotations, with_bbox=True, with_mask=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'instances')
    )
]
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/lvis_v1_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        return_classes=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type=InsClsIoUMetric,
    prefix='lvis_ins',
    base_classes=LVIS_BASE_IDS,
    novel_classes=LVIS_RARE_IDS,
)
test_evaluator = val_evaluator
