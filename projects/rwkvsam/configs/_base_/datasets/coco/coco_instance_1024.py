# dataset settings
from mmcv import LoadImageFromFile, RandomResize
from mmdet.datasets import AspectRatioBatchSampler, CocoDataset
from mmdet.datasets.transforms import LoadAnnotations, Resize, RandomFlip, PackDetInputs, RandomCrop
from mmdet.evaluation import CocoMetric
from mmengine.dataset import DefaultSampler

from seg.datasets.pipeliens.frame_copy import AddSemSeg
from seg.datasets.pipeliens.loading import FilterAnnotationsHB

data_root = 'data/coco/'
backend_args = None
dataset_type = CocoDataset

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
    dict(
        type=AddSemSeg,
    ),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=RandomResize,
        resize_type=Resize,
        scale=image_size,
        ratio_range=(0.1, 2.0),
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
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
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
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
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
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args
)
test_evaluator = val_evaluator
