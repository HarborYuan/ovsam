from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from mmdet.models import DetDataPreprocessor
from mmdet.registry import MODELS
try:
    from kornia.contrib import distance_transform
except ImportError:
    distance_transform = None
from mmengine.structures import InstanceData


def get_center_coords(gt_instances, rescale_shape=None, device='cpu'):
    assert distance_transform is not None
    if rescale_shape is not None:
        masks = gt_instances.masks
        masks = masks.rescale(rescale_shape)
    else:
        masks = gt_instances.masks
    masks = masks.to_tensor(dtype=torch.bool, device=device)[:, None]
    point_coords = []
    for mask in masks:
        mask = mask[None]
        n, _, h, w = mask.shape
        mask_dt = (
            distance_transform(
                (~F.pad(mask, pad=(1, 1, 1, 1), mode='constant', value=0)).float()
            )[:, :, 1:-1, 1:-1]
        )
        selected_point = torch.tensor([mask_dt.argmax() / w, mask_dt.argmax() % w]).long().flip(0).to(
            device)
        point_coords.append(selected_point)
    if len(point_coords) > 0:
        point_coords = torch.stack(point_coords)[:, None]
    else:
        point_coords = torch.empty((0, 1, 2), dtype=torch.int32).to(device=device)
    return point_coords


def get_random_points(gt_instances, device='cpu', num_points=1):
    point_coords = []
    for instance_idx in range(len(gt_instances)):
        mask = gt_instances.masks.masks[instance_idx]
        candidate_indices = torch.tensor(mask, device=device).nonzero()
        assert len(candidate_indices) > 0
        selected_point = candidate_indices[torch.randperm(
            len(candidate_indices), dtype=torch.int32, device=device)[:num_points]].flip(1)
        point_coords.append(selected_point)
    if len(point_coords) > 0:
        point_coords = torch.stack(point_coords)
    else:
        point_coords = torch.empty((0, 1, 2), dtype=torch.int32).to(device=device)
    return point_coords


@MODELS.register_module()
class OVSAMDataPreprocessor(DetDataPreprocessor):
    """
    Open-Vocabulary SAM Preprocessor
    bp : box=0; point=1
    """
    def __init__(self, *args,
                 # Train configs
                 use_det_only: bool = False,
                 use_point_only: bool = False,
                 use_center_point: bool = False,
                 use_point_det: bool = False,
                 use_point_det_more_box: bool = False,
                 use_point_det_pad_mode: bool = False,
                 # Test configs
                 use_img_center: bool = False,
                 use_custom_bbox: Optional[Tuple] = None,
                 use_custom_point: Optional[Tuple] = None,
                 use_gt_box: bool = False,
                 num_proposals: int = 60,
                 default_mode: str = 'sam',
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Training configs
        self.num_proposals = num_proposals
        self.use_det_only = use_det_only
        self.use_point_only = use_point_only
        self.use_center_point = use_center_point
        self.use_point_det = use_point_det
        self.use_point_det_more_box = use_point_det_more_box
        self.use_point_det_pad_mode = use_point_det_pad_mode

        # Eval configs
        self.use_img_center = use_img_center
        self.use_custom_bbox = use_custom_bbox
        self.use_custom_point = use_custom_point
        self.use_gt_box = use_gt_box
        self.default_mode = default_mode

        # Other params
        self.max_points = 5

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super().forward(data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']
        if 'data_tag' in data_samples[0]:
            data_tag = data_samples[0].data_tag
            for i in range(1, len(data_samples)):
                assert data_samples[i].data_tag == data_tag
        else:
            data_tag = self.default_mode
            for i in range(0, len(data_samples)):
                data_samples[i].data_tag = data_tag
        device = inputs.device

        if data_tag == 'sam':
            num_proposals = self.num_proposals if training else 10000000
            if not training and self.use_custom_bbox:
                for data_sample in data_samples:
                    img_shape = data_sample.img_shape
                    data_sample.gt_instances = InstanceData(
                        bboxes=inputs.new_tensor([[img_shape[1] * self.use_custom_bbox[0],
                                                   img_shape[0] * self.use_custom_bbox[1],
                                                   img_shape[1] * self.use_custom_bbox[2],
                                                   img_shape[0] * self.use_custom_bbox[3]]])
                    )
            elif not training and self.use_custom_point:
                for data_sample in data_samples:
                    data_sample.gt_instances = InstanceData(
                        point_coords=inputs.new_tensor([[[self.use_custom_point[0], self.use_custom_point[1]]]])
                    )
            elif not training and self.use_img_center:
                for data_sample in data_samples:
                    data_sample.gt_instances = InstanceData(
                        point_coords=inputs.new_tensor([[[data_sample.img_shape[1] / 2, data_sample.img_shape[0] / 2]]])
                    )
            elif not training and self.use_gt_box:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    if len(gt_instances) > num_proposals:
                        gt_instances = gt_instances[:num_proposals]
                    # Required for testing
                    bboxes = gt_instances.bboxes
                    scale_factor = bboxes.new_tensor(data_sample.scale_factor).repeat(2)
                    bboxes = bboxes * scale_factor
                    gt_instances.bboxes = bboxes
                    # Required for testing
                    num_ins = len(gt_instances)
                    bp_indicator = torch.zeros((num_ins,))
                    gt_instances.bp = bp_indicator.to(device=device)
                    data_sample.gt_instances = gt_instances
            elif training and self.use_det_only:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    if len(gt_instances) > num_proposals:
                        gt_instances = gt_instances[:num_proposals]
                    num_ins = len(gt_instances)
                    bp_indicator = torch.zeros((num_ins,))
                    gt_instances.bp = bp_indicator.to(device=device)
                    data_sample.gt_instances = gt_instances
            elif training and self.use_point_only:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    if len(gt_instances) > num_proposals:
                        gt_instances = gt_instances[:num_proposals]
                    num_points = torch.randint(low=1, high=self.max_points + 1, size=(1,)).item()
                    gt_instances.point_coords = get_random_points(gt_instances, device=device, num_points=num_points)
                    num_ins = len(gt_instances)
                    bp_indicator = torch.ones((num_ins,))
                    gt_instances.bp = bp_indicator.to(device=device)
                    data_sample.gt_instances = gt_instances
            elif training and self.use_point_det:
                if self.use_point_det_more_box:
                    b_or_p = torch.randint(low=-1, high=2, size=(1,)).clamp(min=0).item()  # box 2 : point 1
                else:
                    b_or_p = torch.randint(low=0, high=3, size=(1,)).item() # box 2 : point 1
                is_any_thing = torch.tensor([data_sample.gt_instances.isthing.any() for data_sample in data_samples])
                if not is_any_thing.all():
                    b_or_p = 1
                if b_or_p == 0:
                    for data_sample in data_samples:
                        gt_instances = data_sample.gt_instances
                        # box only supports thing classes
                        gt_instances = gt_instances[gt_instances.isthing]
                        if len(gt_instances) > num_proposals:
                            gt_instances = gt_instances[:num_proposals]
                        num_ins = len(gt_instances)
                        bp_indicator = torch.zeros((num_ins,))
                        gt_instances.bp = bp_indicator.to(device=device)
                        data_sample.gt_instances = gt_instances
                elif b_or_p == 1 or b_or_p == 2:
                    num_points = torch.randint(low=1, high=self.max_points + 1, size=(1,)).item()
                    for data_sample in data_samples:
                        gt_instances = data_sample.gt_instances
                        if len(gt_instances) > num_proposals:
                            gt_instances = gt_instances[:num_proposals]
                        gt_instances.point_coords = get_random_points(gt_instances, device=device, num_points=num_points)
                        num_ins = len(gt_instances)
                        bp_indicator = torch.ones((num_ins,))
                        gt_instances.bp = bp_indicator.to(device=device)
                        data_sample.gt_instances = gt_instances
                else:
                    raise ValueError
            elif self.use_point_det_pad_mode:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    if len(gt_instances) > num_proposals:
                        gt_instances = gt_instances[torch.randperm(len(gt_instances), device=device)]
                        gt_instances = gt_instances[:num_proposals]
                    num_points = torch.randint(low=1, high=self.max_points + 1, size=(1,)).item()
                    gt_instances.point_coords = get_random_points(gt_instances, device=device, num_points=num_points)
                    num_ins = len(gt_instances)
                    bp_indicator = torch.randint(low=-1, high=2, size=(num_ins,)).clamp(min=0)
                    bp_indicator[torch.logical_not(gt_instances.isthing)] = 1
                    gt_instances.bp = bp_indicator.to(device=device)
                    data_sample.gt_instances = gt_instances
            elif training and self.use_center_point:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    gt_instances = gt_instances[:num_proposals]
                    if training:
                        gt_instances.point_coords = get_center_coords(gt_instances, device=device)
                    else:
                        gt_instances.point_coords = get_center_coords(
                            gt_instances, rescale_shape=data_sample.img_shape, device=device
                        )
                    data_sample.gt_instances = gt_instances
            else:
                raise NotImplementedError
        elif data_tag == 'coco':
            pass
        else:
            raise NotImplementedError
        return dict(inputs=inputs, data_samples=data_samples)
