import copy
import os.path as osp
from typing import List

from mmdet.datasets import BaseDetDataset
from mmdet.datasets.api_wrappers import COCOPanoptic
from mmdet.registry import DATASETS
from mmengine.fileio import get_local_path

@DATASETS.register_module()
class CocoNutPanopticDataset(BaseDetDataset):

    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
         'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
         'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
         'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
         'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
         'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
         'wall-wood', 'water-other', 'window-blind', 'window-other',
         'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
         'cabinet-merged', 'table-merged', 'floor-other-merged',
         'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
         'paper-merged', 'food-other-merged', 'building-other-merged',
         'rock-merged', 'wall-other-merged', 'rug-merged'),
        'thing_classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        'stuff_classes':
        ('banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
         'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
         'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
         'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
         'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
         'wall-wood', 'water-other', 'window-blind', 'window-other',
         'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
         'cabinet-merged', 'table-merged', 'floor-other-merged',
         'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
         'paper-merged', 'food-other-merged', 'building-other-merged',
         'rock-merged', 'wall-other-merged', 'rug-merged'),
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208), (255, 255, 128), (147, 211, 203),
         (150, 100, 100), (168, 171, 172), (146, 112, 198), (210, 170, 100),
         (92, 136, 89), (218, 88, 184), (241, 129, 0), (217, 17, 255),
         (124, 74, 181), (70, 70, 70), (255, 228, 255), (154, 208, 0),
         (193, 0, 92), (76, 91, 113), (255, 180, 195), (106, 154, 176),
         (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
         (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
         (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
         (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
         (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
         (146, 139, 141), (70, 130, 180), (134, 199, 156), (209, 226, 140),
         (96, 36, 108), (96, 96, 96), (64, 170, 64), (152, 251, 152),
         (208, 229, 228), (206, 186, 171), (152, 161, 64), (116, 112, 0),
         (0, 114, 143), (102, 102, 156), (250, 141, 255)]
    }
    COCOAPI = COCOPanoptic

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            # changed here for bugfix
            raw_ann_info = self.coco.img_ann_map[img_id]

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``.

        Returns:
            dict: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        # filter out unmatched annotations which have
        # same segment_id but belong to other image
        ann_info = [
            ann for ann in ann_info if ann['image_id'] == img_info['img_id']
        ]
        data_info = {}

        if 'unlabeled2017' in img_info['coco_url']:
            img_path = osp.join(
                self.data_prefix['img'].replace('train2017', 'unlabeled2017'), img_info['file_name']
            )
        else:
            img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].replace('jpg', 'png'))
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['thing_classes']
            data_info['stuff_text'] = self.metainfo['stuff_classes']
            data_info['custom_entities'] = True  # no important

        instances = []
        segments_info = []
        for ann in ann_info:
            instance = {}
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            category_id = ann['category_id']
            contiguous_cat_id = self.cat2label[category_id]

            is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
            assert is_thing == ann['isthing']

            if is_thing:
                is_crowd = ann.get('iscrowd', False)
                instance['bbox'] = bbox
                instance['bbox_label'] = contiguous_cat_id
                if not is_crowd:
                    instance['ignore_flag'] = 0
                else:
                    instance['ignore_flag'] = 1
                    is_thing = False

            segment_info = {
                'id': ann['id'],
                'category': contiguous_cat_id,
                'is_thing': is_thing
            }
            segments_info.append(segment_info)
            if len(instance) > 0 and is_thing:
                instances.append(instance)
        data_info['instances'] = instances
        data_info['segments_info'] = segments_info
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter images too small or without ground truth.

        Returns:
            List[dict]: ``self.data_list`` after filtering.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        ids_with_ann = set()
        # check whether images have legal thing annotations.
        for data_info in self.data_list:
            for segment_info in data_info['segments_info']:
                if not segment_info['is_thing']:
                    continue
                ids_with_ann.add(data_info['img_id'])

        valid_data_list = []
        for data_info in self.data_list:
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_with_ann:
                continue
            if min(width, height) >= min_size:
                valid_data_list.append(data_info)

        return valid_data_list
