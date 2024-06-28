from typing import Optional

import numpy as np

import mmcv
import torch
from mmcv import BaseTransform
from mmdet.datasets.transforms import LoadAnnotations
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.structures.mask import BitmapMasks
from mmengine import fileio

from seg.models.utils import NO_OBJ


@TRANSFORMS.register_module()
class LoadMaskFromFile(BaseTransform):

    def __init__(
            self,
            flag: str = 'unchanged',
            imdecode_backend: str = 'cv2',
            thr: int = 128,
    ):
        self.flag = flag
        self.imdecode_backend = imdecode_backend
        self.thr = thr

    @staticmethod
    def mask2bbox(mask):
        bbox = np.zeros((4,), dtype=np.float32)
        x_any = np.any(mask, axis=0)
        y_any = np.any(mask, axis=1)
        x = np.where(x_any)[0]
        y = np.where(y_any)[0]
        if len(x) > 0 and len(y) > 0:
            bbox = np.array((x[0], y[0], x[-1], y[-1]), dtype=np.float32)
        return bbox

    def transform(self, results: dict) -> Optional[dict]:
        filename = results['ann_path']
        file_client = fileio.FileClient.infer_client(
            None, filename)
        img_bytes = file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, backend=self.imdecode_backend, flag=self.flag)
        assert len(img.shape) == 2
        h, w = img.shape[:2]
        # assert np.all((img == 0) | (img == 255))
        mask = (img > self.thr).astype(np.uint8)
        bbox = self.mask2bbox(mask)

        results['gt_masks'] = BitmapMasks([mask], h, w)

        _, box_type_cls = get_box_type('hbox')
        results['gt_bboxes'] = box_type_cls(bbox[None], dtype=torch.float32)

        results['gt_bboxes_labels'] = np.zeros((1,), dtype=np.int64)
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}'

        return repr_str



@TRANSFORMS.register_module()
class LoadAllPanopticAnnotations(LoadAnnotations):
    # two changes:
    # 1. 255 -> 65535
    # 2. stuff to masks
    def __init__(self,
                 with_bbox: bool = True,
                 with_label: bool = True,
                 with_mask: bool = True,
                 with_seg: bool = True,
                 box_type: str = 'hbox',
                 imdecode_backend: str = 'cv2',
                 backend_args: dict = None) -> None:
        try:
            from panopticapi import utils
        except ImportError:
            raise ImportError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')
        self.rgb2id = utils.rgb2id

        super(LoadAllPanopticAnnotations, self).__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            with_keypoints=False,
            box_type=box_type,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)

    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])

        for instance in results.get('segments_info', []):
            if not instance.get('is_thing'):
                gt_bboxes.append([0, 0, results.get('height'), results.get('width')])
                gt_ignore_flags.append(False)

        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])

        for instance in results.get('segments_info', []):
            if not instance.get('is_thing'):
                gt_bboxes_labels.append(instance['category'])

        # TODO: Inconsistent with mmcv, consider how to deal with it later.
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _load_masks_and_semantic_segs(self, results: dict) -> None:
        """Private function to load mask and semantic segmentation annotations.

        In gt_semantic_seg, the foreground label is from ``0`` to
        ``num_things - 1``, the background label is from ``num_things`` to
        ``num_things + num_stuff - 1``, 255 means the ignored label (``VOID``).

        Args:
            results (dict): Result dict from :obj:``mmdet.CustomDataset``.
        """
        # seg_map_path is None, when inference on the dataset without gts.
        if results.get('seg_map_path', None) is None:
            return

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        pan_png = mmcv.imfrombytes(
            img_bytes, flag='color', channel_order='rgb').squeeze()
        pan_png = self.rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png) + NO_OBJ  # 255 as ignore

        gt_stuff_masks = []
        for segment_info in results['segments_info']:
            mask = (pan_png == segment_info['id'])
            gt_seg = np.where(mask, segment_info['category'], gt_seg)

            # The legal thing masks
            if segment_info.get('is_thing'):
                gt_masks.append(mask.astype(np.uint8))
            else:
                gt_stuff_masks.append(mask.astype(np.uint8))

        gt_masks = gt_masks + gt_stuff_masks
        if self.with_mask:
            h, w = results['ori_shape']
            gt_masks = BitmapMasks(gt_masks, h, w)
            results['gt_masks'] = gt_masks

        if self.with_seg:
            results['gt_seg_map'] = gt_seg

    def transform(self, results: dict) -> dict:
        """Function to load multiple types panoptic annotations.

        Args:
            results (dict): Result dict from :obj:``mmdet.CustomDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask or self.with_seg:
            # The tasks completed by '_load_masks' and '_load_semantic_segs'
            # in LoadAnnotations are merged to one function.
            self._load_masks_and_semantic_segs(results)

        return results
