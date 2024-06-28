from typing import Optional, Sequence, Dict

import numpy as np
import torch
from mmengine.dist import collect_results, broadcast_object_list, is_main_process

from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS
from mmengine.evaluator.metric import _to_cpu

from projects.rwkvsam.utils.boundary_iou import mask_to_boundary


@METRICS.register_module()
class BIoUMetric(BaseMetric):

    def __init__(self, collect_device: str = 'cpu', prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ious = []

    def get_iou(self, gt_masks, pred_masks):
        gt_masks = gt_masks
        n, h, w = gt_masks.shape
        intersection = (gt_masks & pred_masks).reshape(n, h * w).sum(dim=-1)
        union = (gt_masks | pred_masks).reshape(n, h * w).sum(dim=-1)
        ious = (intersection / (union + 1.e-8))
        return ious

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_masks = data_sample['pred_instances']['masks']
            gt_masks = data_sample['gt_instances']['masks']
            device = pred_masks.device

            gt_masks = gt_masks.to_tensor(dtype=torch.bool, device=pred_masks.device)
            gt_boundary = mask_to_boundary(gt_masks.cpu().numpy()[0].astype(np.uint8))
            pred_boundary = mask_to_boundary(pred_masks.cpu().numpy()[0].astype(np.uint8))

            biou = self.get_iou(
                torch.tensor(gt_boundary[None]).to(device=device),
                torch.tensor(pred_boundary[None]).to(device=device)
            )
            self.ious.append(biou)

    def compute_metrics(self, iou_list) -> Dict[str, float]:
        mean_iou = sum(iou_list) / len(iou_list)
        results = dict()
        results['biou'] = mean_iou * 100
        return results

    def evaluate(self, size: int) -> dict:
        _ious = collect_results(self.ious, size, self.collect_device)
        if is_main_process():
            _ious = _to_cpu(_ious)
            ious = torch.cat(_ious)
            _metrics = self.compute_metrics(ious)
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore
        broadcast_object_list(metrics)
        return metrics[0]
