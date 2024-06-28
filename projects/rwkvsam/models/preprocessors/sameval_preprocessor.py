import torch

from mmdet.models import DetDataPreprocessor
from mmdet.registry import MODELS
from mmdet.structures.mask import BitmapMasks


@MODELS.register_module()
class SAMEvalDataPreprocessor(DetDataPreprocessor):

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super().forward(data, training=True)
        for data_sample in data['data_samples']:
            pred_instances = data_sample.pred_instances
            bboxes = pred_instances.bboxes
            scale_factor = bboxes.new_tensor(data_sample.scale_factor).repeat(2)
            bboxes = bboxes * scale_factor
            pred_instances.bboxes = bboxes

            if 'masks' in pred_instances:
                masks = BitmapMasks(pred_instances.masks.to(device='cpu', dtype=torch.uint8).numpy(),
                                    *pred_instances.masks.shape[-2:])
                masks = masks.resize(data_sample.img_shape)
                if self.pad_mask:
                    masks = masks.pad(data_sample.batch_input_shape, pad_val=self.mask_pad_value)
                pred_instances.masks = masks

        return data
