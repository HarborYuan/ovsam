from typing import List

from mmdet.registry import DATASETS
from mmengine import print_log, join_path, list_dir_or_file
from mmdet.datasets import BaseDetDataset

@DATASETS.register_module()
class DIS5KDataset(BaseDetDataset):
    METAINFO = {
        'classes': None,
        'palette': None,
    }

    def __init__(self, *args, img_map_suffix='.jpg', **kwargs):
        self.img_map_suffix = img_map_suffix
        self.id2folder = dict()
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        print_log('Starting to load DIS5K dataset', 'current')
        folders = []
        if 'TR' in self.ann_file:
            folders.append('DIS-TR')
        elif 'VD' in self.ann_file:
            folders.append('DIS-VD')

        img_ids_list = []
        for folder in folders:
            folder_path = join_path(self.data_prefix['img'], folder)
            im_folder_path = join_path(folder_path, 'im')
            img_ids = sorted(
                list_dir_or_file(im_folder_path, recursive=False, list_dir=False, suffix=self.img_map_suffix)
            )
            img_ids_list.extend(img_ids)
            for img_id in img_ids:
                self.id2folder[img_id] = folder

        img_ids = img_ids_list
        data_list = []
        for img_id in img_ids:
            data_info = {
                'img_id': img_id,
                'img_path': join_path(self.data_prefix['img'], self.id2folder[img_id], 'im', img_id),
                'ann_path': join_path(self.data_prefix['img'], self.id2folder[img_id], 'gt',
                                      img_id.replace('.jpg', '.png')),
            }
            data_list.append(data_info)
        print_log(f'Found {len(data_list)} in {len(folders)} folders.', 'current')
        return data_list
