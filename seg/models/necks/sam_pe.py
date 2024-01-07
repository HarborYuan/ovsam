from typing import Tuple, Literal

import torch
from mmengine import MMLogger

from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmengine.structures import InstanceData

from ext.sam import PromptEncoder
from ext.meta.sam_meta import meta_dict, checkpoint_dict
from seg.models.utils.load_checkpoint import load_checkpoint_with_prefix


@MODELS.register_module()
class SAMPromptEncoder(BaseModule):

    def __init__(
            self,
            model_name: Literal['vit_h', 'vit_l', 'vit_b'] = 'vit_h',
            fix: bool = True,
            init_cfg=None,
    ):
        assert init_cfg is not None and init_cfg['type'] == 'sam_pretrain', f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__(init_cfg=None)
        self.init_cfg = init_cfg
        self.logger = MMLogger.get_current_instance()

        backbone_meta = meta_dict[model_name]
        checkpoint_path = checkpoint_dict[pretrained]

        prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(backbone_meta['image_embedding_size'], backbone_meta['image_embedding_size']),
            input_image_size=(backbone_meta['image_size'], backbone_meta['image_size']),
            mask_in_chans=16,
        )
        state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix='prompt_encoder')
        prompt_encoder.load_state_dict(state_dict, strict=True)

        # meta
        self.embed_dim = prompt_encoder.embed_dim
        self.input_image_size = prompt_encoder.input_image_size
        self.image_embedding_size = prompt_encoder.image_embedding_size
        self.num_point_embeddings = 4
        self.mask_input_size = prompt_encoder.mask_input_size

        # positional encoding
        self.pe_layer = prompt_encoder.pe_layer

        # mask encoding
        self.mask_downscaling = prompt_encoder.mask_downscaling
        self.no_mask_embed = prompt_encoder.no_mask_embed

        # point encoding
        self.point_embeddings = prompt_encoder.point_embeddings
        self.not_a_point_embed = prompt_encoder.not_a_point_embed

        self.fix = fix
        if self.fix:
            self.train(mode=False)
            for name, param in self.named_parameters():
                param.requires_grad = False

    @property
    def device(self):
        return self.no_mask_embed.weight.device

    def init_weights(self):
        self.logger.info(f"Init Config for {self.__class__.__name__}")
        self.logger.info(self.init_cfg)

    def train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if self.fix:
            super().train(mode=False)
        else:
            super().train(mode=mode)
        return self

    def _embed_boxes(self, bboxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Embeds box prompts."""
        bboxes = bboxes + 0.5  # Shift to center of pixel
        coords = bboxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def forward(
            self,
            instances: InstanceData,
            image_size: Tuple[int, int],
            with_points: bool = False,
            with_bboxes: bool = False,
            with_masks: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert with_points or with_bboxes or with_masks
        bs = len(instances)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self.device)
        if with_points:
            assert 'point_coords' in instances
            coords = instances.point_coords
            labels = torch.ones_like(coords)[:, :, 0]
            point_embeddings = self._embed_points(coords, labels, pad=not with_bboxes)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if with_bboxes:
            assert 'bboxes' in instances
            box_embeddings = self._embed_boxes(
                instances.bboxes, image_size=image_size
            )
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if with_masks:
            assert 'masks' in instances
            dense_embeddings = self._embed_masks(instances.masks.masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        return sparse_embeddings, dense_embeddings
