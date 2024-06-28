# dataset settings
from mmengine import read_base
from mmengine.dataset import RepeatDataset
from projects.rwkvsam.datasets import AdvancedConcatDataset

with read_base():
    from ..coconut.coconut_b_instance_lsj import train_dataloader as _coconut
    from ..entity.entity_lr_instance_lsj import train_dataloader as _entity
    from ..DIS.dis_5k_1024 import train_dataloader as _dis
    from ..sam.sam_001 import train_dataloader as _sam

    # import tests
    from ..DIS.dis_5k_1024 import *

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    # batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=AdvancedConcatDataset,  # 233960 (5x; 2 : 1 : 1: 1)
        data_tag=['sam', 'sam', 'sam', 'sam'],
        datasets=[
            dict(
                type=RepeatDataset,  # 233960 (2x)
                dataset=_coconut.dataset,
                times=1,
            ),
            dict(
                type=RepeatDataset,  # 31913 (0.27x)
                dataset=_entity.dataset,
                times=4,
            ),
            dict(
                type=RepeatDataset,  # 3000 (0.025x)
                dataset=_dis.dataset,
                times=40,
            ),
            dict(
                type=RepeatDataset,  # 111860 ~1x
                dataset=_sam.dataset,
                times=1,
            ),
        ],
    )
)
