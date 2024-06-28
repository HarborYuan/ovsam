# optimizer
from mmengine.optim import OptimWrapper, PolyLR, LinearLR
from mmengine.runner import IterBasedTrainLoop, ValLoop, TestLoop
from torch.optim import AdamW

optimizer = dict(type=AdamW, lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=optimizer,
    clip_grad=None,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }
    )
)

# learning policy
param_scheduler = [
    dict(type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]
# training schedule for 160k
train_cfg = dict(
    type=IterBasedTrainLoop, max_iters=160000, val_interval=16000
)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)
