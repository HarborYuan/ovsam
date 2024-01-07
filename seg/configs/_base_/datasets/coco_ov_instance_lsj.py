# dataset settings
from mmcv import LoadImageFromFile, RandomResize
from mmdet.datasets import AspectRatioBatchSampler
from mmdet.datasets.transforms import LoadAnnotations, Resize, RandomFlip, PackDetInputs, RandomCrop
from mmengine.dataset import DefaultSampler

from seg.datasets.coco_ins_ov import CocoOVDataset
from seg.datasets.pipeliens.loading import FilterAnnotationsHB

data_root = 'data/coco/'
backend_args = None
dataset_type = CocoOVDataset

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
        ratio_range=(.9, 2.),
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
        filter_cfg=dict(filter_empty_gt=True, min_size=32, sub_split='48_17'),
        pipeline=train_pipeline,
        backend_args=backend_args)
)

val_dataloader = None
test_dataloader = val_dataloader

val_evaluator = None
test_evaluator = val_evaluator
