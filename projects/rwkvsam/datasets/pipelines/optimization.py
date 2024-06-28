from typing import Optional

from mmcv import BaseTransform
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class FixPadOptimization(BaseTransform):

    def __init__(
            self,
            img_scale=None,
            mask_pad_val=0,
            num_proposals=0,
):
        self.img_scale = img_scale
        self.mask_pad_val = mask_pad_val
        self.num_proposals = num_proposals

    def transform(self, results: dict) -> Optional[dict]:
        if self.num_proposals > 0:
            results['data_samples'].gt_instances = results['data_samples'].gt_instances[:self.num_proposals]
        results['data_samples'].gt_instances.masks = results['data_samples'].gt_instances.masks.pad(
            self.img_scale,
            pad_val=self.mask_pad_val)
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}'
        return repr_str
