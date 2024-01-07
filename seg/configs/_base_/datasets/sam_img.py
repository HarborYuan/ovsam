from mmcv import LoadImageFromFile
from mmengine.dataset import DefaultSampler

from seg.datasets.pipeliens.formatting import PackSAMInputs
from seg.datasets.pipeliens.loading import LoadJSONFromFile, LoadFeatFromFile
from seg.datasets.pipeliens.transforms import ResizeSAM
from seg.datasets.sam import SAMDataset
from seg.datasets.pipeliens.loading import LoadAnnotationsSAM

dataset_type = SAMDataset
data_root = 'data/sam'

backend_args = None
image_size = (1024, 1024)

# dataset settings
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=ResizeSAM, scale=image_size, keep_ratio=True),
    dict(type=LoadFeatFromFile),
    dict(
        type=PackSAMInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    ),
]

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadJSONFromFile, backend_args=backend_args),
    dict(type=LoadAnnotationsSAM, with_bbox=True, with_mask=True, with_point_coords=True),
    dict(type=ResizeSAM, scale=image_size, keep_ratio=True),
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
        ann_file='val.txt',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

val_evaluator = []
test_evaluator = val_evaluator
