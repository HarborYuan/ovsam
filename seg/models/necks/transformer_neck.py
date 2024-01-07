from functools import partial
from typing import Tuple, List, Optional

import torch
from torch import Tensor, nn

from mmengine.model import BaseModule, normal_init
from mmdet.registry import MODELS
from mmdet.models.layers import PatchEmbed

from ext.meta.sam_meta import checkpoint_dict
from ext.sam.common import LayerNorm2d
from ext.sam.image_encoder import Block

from seg.models.utils.load_checkpoint import load_checkpoint_with_prefix


@MODELS.register_module()
class MultiLayerTransformerNeck(BaseModule):
    STRIDE = 16

    def __init__(
            self,
            input_size: Tuple[int, int],
            in_channels: List[int],
            embed_channels: int,
            out_channels: int,
            layer_ids: Tuple[int] = (0, 1, 2, 3),
            strides: Tuple[int] = (4, 8, 16, 32),
            embedding_path: Optional[str] = None,
            fix=False,
            init_cfg=None
    ) -> None:
        super().__init__(init_cfg=None)

        self.transformer_size = (input_size[0] // self.STRIDE, input_size[1] // self.STRIDE)
        self.layer_ids = layer_ids

        self.patch_embeds = nn.ModuleList()
        for idx, in_ch in enumerate(in_channels):
            if idx in layer_ids:
                if strides[idx] > self.STRIDE:
                    patch_embed = PatchEmbed(
                        conv_type=nn.ConvTranspose2d,
                        in_channels=in_ch,
                        embed_dims=embed_channels,
                        kernel_size=strides[idx] // self.STRIDE,
                        stride=strides[idx] // self.STRIDE,
                        input_size=(input_size[0] // strides[idx], input_size[1] // strides[idx])
                    )
                else:
                    patch_embed = PatchEmbed(
                        in_channels=in_ch,
                        embed_dims=embed_channels,
                        kernel_size=self.STRIDE // strides[idx],
                        stride=self.STRIDE // strides[idx],
                        input_size=(input_size[0] // strides[idx], input_size[1] // strides[idx])
                    )
                self.patch_embeds.append(patch_embed)
            else:
                self.patch_embeds.append(nn.Identity())

        if embedding_path is not None:
            assert embedding_path.startswith('sam_')
            embedding_ckpt = embedding_path.split('_', maxsplit=1)[1]
            path = checkpoint_dict[embedding_ckpt]
            state_dict = load_checkpoint_with_prefix(path, prefix='image_encoder')
            pos_embed = state_dict['pos_embed']
        else:
            # For loading from checkpoint
            pos_embed = torch.zeros(1, input_size[0] // self.STRIDE, input_size[1] // self.STRIDE, embed_channels)

        self.register_buffer('pos_embed', pos_embed)

        self.level_encoding = nn.Embedding(len(layer_ids), embed_channels)

        depth = 5
        global_attn_indexes = [4]
        window_size = 14

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_channels,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                use_rel_pos=True,
                rel_pos_zero_init=True,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=self.transformer_size,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
        )

        self.fix = fix
        if self.fix:
            self.train(mode=False)
            for name, param in self.named_parameters():
                param.requires_grad = False

        if init_cfg is not None:
            assert init_cfg['type'] == 'Pretrained'
            checkpoint_path = init_cfg['checkpoint']
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=True)
            self._is_init = True

    def init_weights(self):
        normal_init(self.level_encoding, mean=0, std=1)

    def train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if self.fix:
            super().train(mode=False)
        else:
            super().train(mode=mode)
        return self

    def forward(self, inputs: Tuple[Tensor]) -> Tensor:
        input_embeddings = []
        level_cnt = 0
        for idx, feat in enumerate(inputs):
            if idx not in self.layer_ids:
                continue
            feat, size = self.patch_embeds[idx](feat)
            feat = feat.unflatten(1, size)
            feat = feat + self.level_encoding.weight[level_cnt]
            input_embeddings.append(feat)
            level_cnt += 1

        feat = sum(input_embeddings)
        feat = feat + self.pos_embed
        for block in self.blocks:
            feat = block(feat)
        feat = feat.permute(0, 3, 1, 2).contiguous()
        feat = self.neck(feat)
        return feat


@MODELS.register_module()
class SingleLayerTransformerNeck(BaseModule):
    STRIDE = 16

    def __init__(
            self,
            input_size: Tuple[int, int],
            in_channels: List[int],
            embed_channels: int,
            out_channels: int,
            layer_id: int = 2,
            strides: Tuple[int] = (4, 8, 16, 32),
            embedding_path: Optional[str] = None,
            init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=None)
        self.layer_id = layer_id
        self.transformer_size = (input_size[0] // self.STRIDE, input_size[1] // self.STRIDE)

        self.patch_embeds = PatchEmbed(
            in_channels=in_channels[self.layer_id],
            embed_dims=embed_channels,
            kernel_size=self.STRIDE // strides[self.layer_id],
            stride=self.STRIDE // strides[self.layer_id],
            input_size=(input_size[0] // strides[self.layer_id], input_size[1] // strides[self.layer_id])
        )

        if embedding_path is not None:
            assert embedding_path.startswith('sam_')
            embedding_ckpt = embedding_path.split('_', maxsplit=1)[1]
            path = checkpoint_dict[embedding_ckpt]
            state_dict = load_checkpoint_with_prefix(path, prefix='image_encoder')
            pos_embed = state_dict['pos_embed']
        else:
            # For loading from checkpoint
            pos_embed = torch.zeros(1, input_size[0] // self.STRIDE, input_size[1] // self.STRIDE, embed_channels)

        self.register_buffer('pos_embed', pos_embed)

        depth = 5
        global_attn_indexes = [4]
        window_size = 14

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_channels,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                use_rel_pos=True,
                rel_pos_zero_init=True,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=self.transformer_size,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
        )

        if init_cfg is not None:
            assert init_cfg['type'] == 'Pretrained'
            checkpoint_path = init_cfg['checkpoint']
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=True)
            self._is_init = True

    def forward(self, inputs: Tuple[Tensor]) -> Tensor:
        feat = inputs[self.layer_id]
        feat, size = self.patch_embeds(feat)
        feat = feat.unflatten(1, size)
        feat = feat + self.pos_embed
        for block in self.blocks:
            feat = block(feat)
        feat = feat.permute(0, 3, 1, 2).contiguous()
        feat = self.neck(feat)
        return feat
