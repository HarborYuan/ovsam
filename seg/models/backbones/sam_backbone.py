from functools import partial
from typing import Literal

import torch
import torch.nn as nn
from mmdet.registry import MODELS

from mmengine.model import BaseModule
from mmengine.logging import MMLogger

from ext.sam import ImageEncoderViT
from ext.meta.sam_meta import meta_dict, checkpoint_dict
from seg.models.utils.load_checkpoint import load_checkpoint_with_prefix


@MODELS.register_module()
class SAMBackbone(BaseModule):

    def __init__(
            self,
            model_name: Literal['vit_h', 'vit_l', 'vit_b'] = 'vit_h',
            fix: bool = True,
            init_cfg=None,
    ):
        assert init_cfg is not None and init_cfg['type'] in \
               ['sam_pretrain', 'Pretrained'], f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__(init_cfg=None)
        self.init_cfg = init_cfg
        self.logger = MMLogger.get_current_instance()

        backbone_meta = meta_dict[model_name]

        backbone = ImageEncoderViT(
            depth=backbone_meta['encoder_depth'],
            embed_dim=backbone_meta['encoder_embed_dim'],
            num_heads=backbone_meta['encoder_num_heads'],
            patch_size=backbone_meta['vit_patch_size'],
            img_size=backbone_meta['image_size'],
            global_attn_indexes=backbone_meta['encoder_global_attn_indexes'],
            out_chans=backbone_meta['prompt_embed_dim'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            qkv_bias=True,
            use_rel_pos=True,
            mlp_ratio=4,
            window_size=14,
        )
        if self.init_cfg['type'] == 'sam_pretrain':
            checkpoint_path = checkpoint_dict[pretrained]
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix='image_encoder')
            backbone.load_state_dict(state_dict, strict=True)

        self.stem = backbone.patch_embed
        self.pos_embed = backbone.pos_embed

        self.res_layers = []
        last_pos = 0
        for idx, cur_pos in enumerate(backbone_meta['encoder_global_attn_indexes']):
            blocks = backbone.blocks[last_pos:cur_pos + 1]
            layer_name = f'layer{idx + 1}'
            self.add_module(layer_name, nn.Sequential(*blocks))
            self.res_layers.append(layer_name)
            last_pos = cur_pos + 1

        self.out_proj = backbone.neck

        if self.init_cfg['type'] == 'Pretrained':
            checkpoint_path = pretrained
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=self.init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=True)

        self.model_name = model_name
        self.fix = fix
        self.model_type = 'vit'
        self.output_channels = None
        self.out_indices = (0, 1, 2, 3)
        if self.fix:
            self.train(mode=False)
            for name, param in self.named_parameters():
                param.requires_grad = False

    def init_weights(self):
        self.logger.info(f"Init Config for {self.model_name}")
        self.logger.info(self.init_cfg)

    def train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if self.fix:
            super().train(mode=False)
        else:
            super().train(mode=mode)
        return self

    def forward_func(self, x):
        x = self.stem(x)
        x = x + self.pos_embed
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x.permute(0, 3, 1, 2).contiguous())
        outs[-1] = self.out_proj(outs[-1])
        return tuple(outs)

    def forward(self, x):
        if self.fix:
            with torch.no_grad():
                outs = self.forward_func(x)
        else:
            outs = self.forward_func(x)
        return outs
