# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
from mmengine.optim import OptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.runner import ValLoop, TestLoop, EpochBasedTrainLoop
from torch.optim import AdamW

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=5e-4 * 1024 / 512,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type=LinearLR,
        start_factor=1e-3,
        by_epoch=True,
        end=10,
        # update by iter
        convert_to_iter_based=True
    ),
    # main learning rate scheduler
    dict(type=CosineAnnealingLR, eta_min=1e-5, by_epoch=True, begin=10)
]

# train, val, test setting
train_cfg = dict(
    type=EpochBasedTrainLoop,
    max_epochs=120,
    val_interval=1,
    dynamic_intervals=[
        (10, 10),
        (100, 1),
    ]
)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(enable=True, base_batch_size=1024)
