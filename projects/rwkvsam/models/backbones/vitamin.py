from functools import partial
from typing import Literal, Tuple, Optional

import math

import torch
from mmcv.cnn import build_activation_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model.weight_init import trunc_normal_
from mmpretrain.models import build_norm_layer, MultiheadAttention, resize_pos_embed
from torch import nn
from torch.utils.checkpoint import checkpoint
from mmengine import MODELS
from mmengine.model import BaseModule

from timm.models import checkpoint_seq
from timm.models.layers import get_norm_act_layer, create_conv2d, make_divisible
from timm.layers.norm_act import _create_act
from timm.layers import DropPath, get_norm_layer


class Downsample2d(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            bias: bool = True,
    ):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        if dim != dim_out:
            self.expand = nn.Conv2d(dim, dim_out, 1, bias=bias)  # 1x1 conv
        else:
            self.expand = nn.Identity()

    def forward(self, x):
        x = self.pool(x)  # spatial downsample
        x = self.expand(x)  # expand chs
        return x


class StridedConv(nn.Module):
    """ downsample 2d as well
    """

    def __init__(
            self,
            kernel_size=3,
            stride=2,
            padding=1,
            in_chans=3,
            embed_dim=768,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        norm_layer = partial(get_norm_layer('layernorm2d'), eps=1e-6)
        self.norm = norm_layer(in_chans)  # affine over C

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        return x


class Stem(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            act_layer: str = 'gelu',
            norm_layer: str = 'layernorm2d',
            norm_eps: float = 1e-6,
            bias: bool = True,
    ):
        super().__init__()
        self.grad_checkpointing = False
        norm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)
        self.out_chs = out_chs
        self.conv1 = create_conv2d(in_chs, out_chs, 3, stride=2, bias=bias)
        self.norm1 = norm_act_layer(out_chs)
        self.conv2 = create_conv2d(out_chs, out_chs, 3, stride=1, bias=bias)

    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                fan_out //= module.groups
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if self.grad_checkpointing:
            x = checkpoint(self.conv1, x)
            x = self.norm1(x)
            x = checkpoint(self.conv2, x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.conv2(x)

        return x


class MbConvLNBlock(nn.Module):
    """ Pre-Norm Conv Block - 1x1 - kxk - 1x1, w/ inverted bottleneck (expand)
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            drop_path: float = 0.,
            kernel_size: int = 3,
            norm_layer: str = 'layernorm2d',
            norm_eps: float = 1e-6,
            act_layer: str = 'gelu',
            expand_ratio: float = 4.0,
    ):
        super(MbConvLNBlock, self).__init__()
        self.stride, self.in_chs, self.out_chs = stride, in_chs, out_chs
        mid_chs = make_divisible(out_chs * expand_ratio)
        prenorm_act_layer = partial(get_norm_act_layer(norm_layer, act_layer), eps=norm_eps)

        if stride == 2:
            self.shortcut = Downsample2d(in_chs, out_chs, bias=True)
        elif in_chs != out_chs:
            self.shortcut = nn.Conv2d(in_chs, out_chs, 1, bias=True)
        else:
            self.shortcut = nn.Identity()

        self.pre_norm = prenorm_act_layer(in_chs, apply_act=False)
        self.down = nn.Identity()
        self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=1, bias=True)
        self.act1 = _create_act(act_layer, inplace=True)
        self.act2 = _create_act(act_layer, inplace=True)

        self.conv2_kxk = create_conv2d(mid_chs, mid_chs, kernel_size, stride=stride, dilation=1, groups=mid_chs,
                                       bias=True)
        self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                fan_out //= module.groups
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.pre_norm(x)
        x = self.down(x)  # nn.Identity()

        # 1x1 expansion conv & act
        x = self.conv1_1x1(x)
        x = self.act1(x)

        # (strided) depthwise 3x3 conv & act
        x = self.conv2_kxk(x)
        x = self.act2(x)

        # 1x1 linear projection to output width
        x = self.conv3_1x1(x)
        x = self.drop_path(x) + shortcut

        return x


class MbConvStages(nn.Module):
    """ MobileConv for stage 1 and stage 2 of ViTamin
    """

    CONFIG = dict(
        # ---
        tiny=dict(
            embed_dim=(32, 64, 192),
            depths=(2, 4, 1),
            stem_width=32,
        ),
        # ---
        small=dict(
            embed_dim=(64, 128, 384),
            depths=(2, 4, 1),
            stem_width=64,
        ),
        base=dict(
            embed_dim=(128, 256, 768),
            depths=(2, 4, 1),
            stem_width=128,
        ),
        large=dict(
            embed_dim=(160, 320, 1024),
            depths=(2, 4, 1),
            stem_width=160,
        )
    )

    def __init__(
            self,
            in_chans: int = 3,
            scale: Literal['tiny', 'small', 'base', 'large'] = 'small'
    ):
        super().__init__()
        self.grad_checkpointing = False
        cfg = self.CONFIG[scale]
        self.stem = Stem(
            in_chs=in_chans,
            out_chs=cfg['stem_width'],
        )
        stages = []
        self.num_stages = len(cfg['embed_dim'])
        for s, dim in enumerate(cfg['embed_dim'][:2]):  # stage
            blocks = []
            stage_in_chs = cfg['embed_dim'][s - 1] if s > 0 else cfg['stem_width']
            for d in range(cfg['depths'][s]):
                blocks += [MbConvLNBlock(
                    in_chs=stage_in_chs if d == 0 else dim,
                    out_chs=dim,
                    stride=2 if d == 0 else 1,
                )]
            blocks = nn.Sequential(*blocks)
            stages += [blocks]

        self.stages = nn.ModuleList(stages)
        self.pool = StridedConv(
            stride=2,
            in_chans=cfg['embed_dim'][1],
            embed_dim=cfg['embed_dim'][2]
        )

        self.embed_dims = cfg['embed_dim'][2]

    def forward(self, x):
        outs = []
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for stage in self.stages:
                x = checkpoint_seq(stage, x)
                outs.append(x)
            x = checkpoint(self.pool, x)
            outs.append(x)
        else:
            for stage in self.stages:
                x = stage(x)
                outs.append(x)
            x = self.pool(x)
            outs.append(x)

        return outs

class GeGluMlp(nn.Module):
    def __init__(
            self,
            embed_dims,
            feedforward_channels,
            act_cfg=None,
            dropout_layer=None,
    ):
        super().__init__()
        self.act = build_activation_layer(act_cfg)
        self.w0 = nn.Linear(embed_dims, feedforward_channels)
        self.w1 = nn.Linear(embed_dims, feedforward_channels)
        self.w2 = nn.Linear(feedforward_channels, embed_dims)

        self.out_drop = build_dropout(dropout_layer)

    def forward(self, x):
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.out_drop(self.w2(x))
        return x


class TransformerLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 layer_scale_init_value=0.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value)

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)


        self.ffn = GeGluMlp(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            act_cfg=act_cfg,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        )

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class RWKVBlock(BaseModule):

    def __init__(
            self,
            channels,
            mlp_ratio=4.,
            drop_path=0.,
            # Meta
            total_layers=None,
            layer_id=None,
            **kwargs
    ):
        super().__init__(init_cfg=None)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        from ext.rwkv.cls_backbones.backbones.vrwkv import VRWKV_SpatialMix, VRWKV_ChannelMix
        self.attn = VRWKV_SpatialMix(
            channels,
            n_layer=total_layers,
            layer_id=layer_id,
            shift_mode='q_shift',
            channel_gamma=.25,
            shift_pixel=1,
            init_mode='fancy',
            key_norm=False,
        )
        self.mlp = VRWKV_ChannelMix(
            channels,
            n_layer=total_layers,
            layer_id=layer_id,
            shift_mode='q_shift',
            channel_gamma=.25,
            shift_pixel=1,
            hidden_rate=mlp_ratio,
            init_mode='fancy',
            key_norm=False
        )
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        """Forward function."""

        # B, N, C = x.shape
        # x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x), (H, W)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x), (H, W)))
        # x = x.view(B, C, N).permute(0, 2, 1)
        return x


class MBConv1d(MbConvLNBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = super().forward(x)
        x = x.flatten(2).transpose(1, 2)
        return x

@MODELS.register_module()
class VITAMINBackbone(BaseModule):
    def __init__(
            self,
            model_variant: Literal['tiny', 'small', 'base', 'large'] = 'small',
            img_size: Tuple[int, int] = (1024, 1024),
            attn_type='transformer',
            attn_cfg:Optional[dict]=None,
            with_pos_embd: bool = True,
            norm_cfg=dict(type='LN', eps=1e-6),
            additional_final_layers: bool = False,
            fix: bool = False,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.fix = fix
        self.model_variant = model_variant
        self.interpolate_mode = 'bicubic'
        self.with_pos_embd = with_pos_embd

        if attn_cfg is None:
            attn_cfg = dict()

        conv_layers = MbConvStages(scale=self.model_variant)
        self.stem = conv_layers.stem
        self.conv_stages = conv_layers.stages
        self.conv_final_pool = conv_layers.pool
        self.embed_dims = conv_layers.embed_dims

        if self.with_pos_embd:
            self.patch_resolution = (img_size[0] // 16, img_size[1] // 16)
            num_patches = self.patch_resolution[0] * self.patch_resolution[1]
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
        else:
            self.patch_resolution = None
            self.pos_embed = None

        self.layers = []
        if attn_type=='transformer':
            depth = attn_cfg.get('depth', 14)
            num_heads = attn_cfg.get('num_heads', 6)
            mlp_ratio = attn_cfg.get('mlp_ratio', 2)
            drop_path_rate = attn_cfg.get('drop_path', 0.)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            for i in range(depth):
                _layer_cfg = dict(
                    embed_dims=self.embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=int(mlp_ratio * self.embed_dims),
                    drop_path_rate=dpr[i]
                )
                self.layers.append(TransformerLayer(**_layer_cfg))
            self.input_patch_res = [False] * depth
        elif attn_type=='rwkv':
            depth = attn_cfg.get('depth', 14)
            mlp_ratio = attn_cfg.get('mlp_ratio', 4)
            drop_path_rate = attn_cfg.get('drop_path', 0.)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            for i in range(depth):
                _layer_cfg = dict(
                    channels=self.embed_dims,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    total_layers=depth,
                    layer_id=i,
                )
                self.layers.append(RWKVBlock(**_layer_cfg))
            self.input_patch_res = [True] * depth
        elif attn_type=='rwkv_trans':
            global_attn_indexes = attn_cfg.get('depth', (2, 5, 8, 11))
            depth = attn_cfg.get('depth', 14)
            mlp_ratio = attn_cfg.get('mlp_ratio', 4)
            num_heads = attn_cfg.get('num_heads', 6)
            drop_path_rate = attn_cfg.get('drop_path', 0.)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            self.input_patch_res = []
            for i in range(depth):
                if i in global_attn_indexes:
                    _layer_cfg = dict(
                        embed_dims=self.embed_dims,
                        num_heads=num_heads,
                        feedforward_channels=int(mlp_ratio * self.embed_dims),
                        drop_path_rate=dpr[i]
                    )
                    self.layers.append(TransformerLayer(**_layer_cfg))
                    self.input_patch_res.append(False)
                else:
                    _layer_cfg = dict(
                        channels=self.embed_dims,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[i],
                        total_layers=depth,
                        layer_id=i,
                    )
                    self.layers.append(RWKVBlock(**_layer_cfg))
                    self.input_patch_res.append(True)
        elif attn_type in ['mbconv_rwkv', 'rwkv_mbconv']:
            if attn_type=='rwkv_mbconv':
                reverse_global = True
            else:
                reverse_global = False
            global_attn_indexes = attn_cfg.get('depth', (2, 5, 8, 11))
            depth = attn_cfg.get('depth', 14)
            mlp_ratio = attn_cfg.get('mlp_ratio', 4)
            drop_path_rate = attn_cfg.get('drop_path', 0.)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            self.input_patch_res = []
            for i in range(depth):
                if bool(i in global_attn_indexes) != reverse_global:
                    _layer_cfg = dict(
                        channels=self.embed_dims,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[i],
                        total_layers=depth,
                        layer_id=i,
                    )
                    self.layers.append(RWKVBlock(**_layer_cfg))
                    self.input_patch_res.append(True)
                else:
                    _layer_cfg = dict(
                        in_chs=self.embed_dims,
                        out_chs=self.embed_dims,
                        stride=1,
                        drop_path=dpr[i],
                        kernel_size= 3,
                        norm_layer= 'layernorm2d',
                        norm_eps= 1e-6,
                        act_layer= 'gelu',
                        expand_ratio=mlp_ratio,
                    )
                    self.layers.append(MBConv1d(**_layer_cfg))
                    self.input_patch_res.append(True)
        else:
            raise NotImplemented
        self.layers = nn.ModuleList(self.layers)
        self.final_norm = build_norm_layer(norm_cfg, self.embed_dims)

        if additional_final_layers:
            self.final_layer = MbConvLNBlock(
                in_chs=self.embed_dims,
                out_chs=self.embed_dims * 2,
                stride=2,
            )
        else:
            self.final_layer = None

    def init_weights(self):
        super().init_weights()

        if not (isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)


    def forward(self, x):
        x = self.stem(x)
        outs = []
        for conv_stage in self.conv_stages:
            x = conv_stage(x)
            outs.append(x)
        x = self.conv_final_pool(x)
        # outs.append(x)

        patch_resolution = x.shape[-2:]
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        if self.with_pos_embd:
            x = x + resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0)

        for idx, layer in enumerate(self.layers):
            if self.input_patch_res[idx]:
                x = layer(x, *patch_resolution)
            else:
                x = layer(x)
            if idx == len(self.layers) // 2 and self.final_layer is None:
                outs.append(x.permute(0, 2, 1).unflatten(2, patch_resolution).contiguous())

        x = self.final_norm(x)

        x = x.permute(0, 2, 1).unflatten(2, patch_resolution).contiguous()
        outs.append(x)
        if self.final_layer is not None:
            x = self.final_layer(x)
            outs.append(x)
        return tuple(outs)
