# optimizer
from mmengine.optim import OptimWrapper, PolyLR
from mmengine.runner import IterBasedTrainLoop, ValLoop, TestLoop
from torch.optim import SGD

optimizer = dict(type=SGD, lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type=OptimWrapper, optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type=PolyLR,
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=160000,
        by_epoch=False
    )
]
# training schedule for 160k
train_cfg = dict(
    type=IterBasedTrainLoop, max_iters=160000, val_interval=16000
)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)
