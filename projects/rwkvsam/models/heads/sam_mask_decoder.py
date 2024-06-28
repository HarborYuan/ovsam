from typing import Literal, Tuple, List, Optional

import torch
from mmcv.ops import point_sample
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.structures import SampleList
from mmdet.utils import reduce_mean
from mmengine import MMLogger
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmengine.structures import InstanceData
from torch import nn

from ext.sam import MaskDecoder
from ext.meta.sam_meta import meta_dict, checkpoint_dict
from projects.rwkvsam.utils.load_checkpoint import load_checkpoint_with_prefix


@MODELS.register_module()
class SAMMaskDecoder(BaseModule):

    def __init__(
            self,
            model_name: Literal['vit_h', 'vit_l', 'vit_b', 'efficient'] = 'vit_h',
            fix: bool = True,
            convert_single: bool = False,
            init_cfg=None,
            loss_mask=None,
            loss_dice=None,
    ):
        assert init_cfg is not None and \
               init_cfg['type'] in ['sam_pretrain', 'Pretrained'], f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__(init_cfg=None)
        self.init_cfg = init_cfg
        self.logger = MMLogger.get_current_instance()
        self.convert_single = convert_single
        self.model_name = model_name

        if model_name == 'efficient':
            from ext.efficient_sam.efficient_sam import MaskDecoder as EfficientSAMDecoder
            from ext.efficient_sam.two_way_transformer import TwoWayTransformer as TransFormer
            mask_decoder = EfficientSAMDecoder(
                transformer_dim=256,
                transformer=TransFormer(
                    depth=2,
                    embedding_dim=256,
                    num_heads=8,
                    mlp_dim=2048,
                    activation=nn.GELU,
                    normalize_before_activation=False,
                ),
                num_multimask_outputs=3,
                activation=nn.GELU,
                normalization_type="layer_norm",
                normalize_before_activation=False,
                iou_head_depth=3 - 1,
                iou_head_hidden_dim=256,
                upscaling_layer_dims=[64, 32],
            )
        else:
            mask_decoder = MaskDecoder(
                num_multimask_outputs=3,
                transformer_dim=meta_dict[model_name]['prompt_embed_dim'],
                iou_head_depth=3,
                iou_head_hidden_dim=256,
            )
        if self.init_cfg['type'] == 'sam_pretrain':
            if pretrained in checkpoint_dict:
                checkpoint_path = checkpoint_dict[pretrained]
            else:
                checkpoint_path = pretrained
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix='mask_decoder')
            mask_decoder.load_state_dict(state_dict, strict=True)

        self.mask_decoder = mask_decoder

        # meta
        self.num_points = 12544
        self.oversample_ratio = 3.
        self.importance_sample_ratio = .75
        if self.convert_single:
            self.query_select = range(1)
            self.num_masks = len(self.query_select)
            # modify mask_decoder
            self.mask_decoder.output_hypernetworks_mlps = self.mask_decoder.output_hypernetworks_mlps[0:1]
        else:
            self.num_masks = mask_decoder.num_mask_tokens

        # losses
        self.loss_dice = MODELS.build(loss_dice) if loss_dice is not None else None
        self.loss_mask = MODELS.build(loss_mask) if loss_mask is not None else None

        if self.init_cfg['type'] == 'Pretrained':
            checkpoint_path = pretrained
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=self.init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=True)

        self.fix = fix
        if self.fix:
            self.train(mode=False)
            for name, param in self.named_parameters():
                param.requires_grad = False

    def init_weights(self):
        self.logger.info(f"Init Config for {self.__class__.__name__}")
        self.logger.info(self.init_cfg)

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        num_instances = int(sparse_prompt_embeddings.shape[0])
        # Concatenate output tokens
        if self.convert_single:
            output_tokens = torch.cat(
                [self.mask_decoder.iou_token.weight, self.mask_decoder.mask_tokens.weight[self.query_select]], dim=0)
        else:
            output_tokens = torch.cat([self.mask_decoder.iou_token.weight, self.mask_decoder.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(num_instances, -1, -1)
        queries = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # image_embeddings = torch.repeat_interleave(image_embeddings, num_instances, dim=0)
        if dense_prompt_embeddings is not None:
            image_embeddings = image_embeddings + dense_prompt_embeddings
        pos_img = torch.repeat_interleave(image_pe, num_instances, dim=0)
        b, c, h, w = image_embeddings.shape

        # Run the transformer
        queries, mask_feats = self.mask_decoder.transformer(image_embeddings, pos_img, queries)
        iou_query = queries[:, 0, :]
        # mask_embeds = queries[:, 1:(1 + self.mask_decoder.num_mask_tokens), :]
        mask_embeds = queries[:, 1:(1 + self.num_masks), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        mask_feats = mask_feats.transpose(1, 2).view(b, c, h, w)
        if self.model_name == 'efficient':
            for upscaling_layer in self.mask_decoder.final_output_upscaling_layers:
                mask_feats = upscaling_layer(mask_feats)
        else:
            mask_feats = self.mask_decoder.output_upscaling(mask_feats)
        mask_queries_list: List[torch.Tensor] = []
        # for i in range(self.mask_decoder.num_mask_tokens):
        for i in range(self.num_masks):
            mask_queries_list.append(self.mask_decoder.output_hypernetworks_mlps[i](mask_embeds[:, i, :]))
        mask_queries = torch.stack(mask_queries_list, dim=1)
        b, c, h, w = mask_feats.shape
        masks = (mask_queries @ mask_feats.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.mask_decoder.iou_prediction_head(iou_query)

        return masks, iou_pred

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: Optional[torch.Tensor],
            multi_mask_output: bool,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_prompts = len(sparse_prompt_embeddings)
        image_embeddings = torch.repeat_interleave(image_embeddings, num_prompts, dim=0)
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multi_mask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def forward_train(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: Optional[torch.Tensor],
            batch_ind_list: List[int],
            data_samples: SampleList,
            **kwargs,
    ):
        image_embed_list = []
        for idx, num_ins in enumerate(batch_ind_list):
            image_embed_list.append(torch.repeat_interleave(image_embeddings[idx:idx + 1], num_ins, dim=0))

        image_embed = torch.cat(image_embed_list, dim=0)
        masks, iou_preds = self.predict_masks(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        instances = []
        for data_sample in data_samples:
            instances.append(InstanceData(
                labels=data_sample.gt_instances.labels,
                masks=data_sample.gt_instances.masks
            ))
        gt_instances = InstanceData.cat(instances)
        assert len(gt_instances) == len(image_embed)

        device = image_embed.device

        pred_masks = masks[:, 0:1]
        gt_masks = gt_instances.masks.to_tensor(dtype=torch.float, device=device)[:, None]

        with torch.no_grad():
            uncertain_points = get_uncertain_point_coords_with_randomness(
                pred_masks, None, self.num_points, self.oversample_ratio, self.importance_sample_ratio)
        pred_masks = point_sample(pred_masks, uncertain_points)
        gt_masks = point_sample(gt_masks, uncertain_points)
        num_masks = len(pred_masks)
        mask_avg_factor = reduce_mean(masks.new_tensor([num_masks]))
        mask_avg_factor = mask_avg_factor.clamp(min=1)
        loss_dice = self.loss_dice(
            pred_masks, gt_masks, avg_factor=mask_avg_factor
        )

        pred_masks = pred_masks.reshape(-1)
        gt_masks = gt_masks.reshape(-1)
        loss_mask = self.loss_mask(
            pred_masks,
            gt_masks,
            avg_factor=mask_avg_factor * self.num_points
        )

        loss_dict = dict()
        loss_dict['loss_dice'] = loss_dice
        loss_dict['loss_mask'] = loss_mask
        loss_dict['loss_zero'] = 0 * iou_preds.sum()
        return loss_dict
