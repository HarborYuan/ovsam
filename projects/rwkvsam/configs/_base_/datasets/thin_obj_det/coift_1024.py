from mmcv import LoadImageFromFile
from mmdet.datasets.transforms import PackDetInputs
from mmengine.dataset import DefaultSampler
from projects.rwkvsam.datasets import ThinOBJDataset
from projects.rwkvsam.datasets.pipelines import LoadMaskFromFile

from seg.datasets.pipeliens.transforms import ResizeSAM

dataset_type = ThinOBJDataset
data_root = 'data/thin_object_detection/COIFT/'

backend_args = None
image_size = (1024, 1024)

# dataset settings
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadMaskFromFile,),
    dict(type=ResizeSAM, scale=image_size, keep_ratio=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    ),
]

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=ResizeSAM, scale=image_size, keep_ratio=True),
    dict(type=LoadMaskFromFile, ),
    dict(
        type=PackDetInputs,
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
        data_prefix=dict(img='images', ann='masks'),
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
        data_prefix=dict(img='images', ann='masks'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

val_evaluator = []
test_evaluator = val_evaluator
