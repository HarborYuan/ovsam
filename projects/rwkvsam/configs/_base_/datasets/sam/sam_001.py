from mmcv import LoadImageFromFile, RandomResize
from mmengine.dataset import DefaultSampler
from mmdet.datasets.transforms import Resize, RandomFlip, PackDetInputs, RandomCrop

from seg.datasets.pipeliens.formatting import PackSAMInputs
from seg.datasets.pipeliens.loading import FilterAnnotationsHB, LoadJSONFromFile, LoadAnnotationsSAM
from seg.datasets.pipeliens.transforms import ResizeSAM

from projects.rwkvsam.datasets import SAMDataset


dataset_type = SAMDataset
data_root = 'data/sam'

backend_args = None
image_size = (1024, 1024)

# dataset settings
train_pipeline = [
    dict(
        type=LoadImageFromFile,
        to_float32=True,
        backend_args=backend_args),
    dict(type=LoadJSONFromFile, backend_args=backend_args, limit=30, max_ratio=1/64),
    dict(type=LoadAnnotationsSAM, with_bbox=True, with_mask=True, with_point_coords=True),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=RandomResize,
        resize_type=Resize,
        scale=image_size,
        ratio_range=(1., 1.5),
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

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=ResizeSAM, scale=image_size, keep_ratio=True),
    dict(type=LoadJSONFromFile, backend_args=backend_args),
    dict(type=LoadAnnotationsSAM, with_bbox=True, with_mask=True, with_point_coords=True),
    dict(
        type=PackSAMInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(img=''),
        filter_cfg=None,
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type=DefaultSampler, shuffle=False, round_up=False),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

val_evaluator = []
test_evaluator = val_evaluator
