from typing import List

from mmdet.registry import DATASETS
from mmengine import print_log, join_path, list_dir_or_file
from mmdet.datasets import BaseDetDataset


@DATASETS.register_module()
class ThinOBJDataset(BaseDetDataset):
    METAINFO = {
        'classes': None,
        'palette': None,
    }

    def __init__(self, *args, img_map_suffix='.jpg', **kwargs):
        self.img_map_suffix = img_map_suffix
        self.id2folder = dict()
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        print_log(f'Starting to load Thin Obj Detection dataset from {self.data_root}', 'current')

        img_ids_list = []
        img_ids = sorted(
            list_dir_or_file(self.data_prefix['img'], recursive=False, list_dir=False, suffix=self.img_map_suffix)
        )
        img_ids_list.extend(img_ids)

        img_ids = img_ids_list
        data_list = []
        for img_id in img_ids:
            data_info = {
                'img_id': img_id,
                'img_path': join_path(self.data_prefix['img'], img_id),
                'ann_path': join_path(self.data_prefix['ann'], img_id.replace('.jpg', '.png')),
            }
            data_list.append(data_info)
        print_log(f'Found {len(data_list)} samples.', 'current')
        return data_list
