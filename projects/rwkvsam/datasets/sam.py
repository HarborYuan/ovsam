from typing import List

from mmdet.registry import DATASETS
from mmengine import get_local_path, list_from_file, join_path, list_dir_or_file, print_log

from mmdet.datasets import BaseDetDataset


@DATASETS.register_module()
class SAMDataset(BaseDetDataset):

    def __init__(self, *args, img_map_suffix='.jpg', custom_structure=False, **kwargs):
        self.img_map_suffix = img_map_suffix
        self.id2folder = dict()
        self.custom_structure = custom_structure
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        print_log('Starting to load sam dataset', 'current')
        if 'sa_1b' in self.ann_file:
            if 'sa_1b_one' in self.ann_file:
                folders = [f'sa_000{str(idx).zfill(3)}' for idx in range(1)]
            elif 'sa_1b_001' in self.ann_file:
                # sa_000000 -> sa_000009
                folders = [f'sa_000{str(idx).zfill(3)}' for idx in range(10)]
            elif 'sa_1b_01' in self.ann_file:
                folders = [f'sa_000{str(idx).zfill(3)}' for idx in range(100)]
            else:
                # sa_000000 -> sa_000999
                folders = [f'sa_000{str(idx).zfill(3)}' for idx in range(1000)]
        else:
            with get_local_path(
                    self.ann_file, backend_args=self.backend_args) as local_path:
                folders = list_from_file(local_path)


        img_ids_list = []
        for folder in folders:
            folder_path = join_path(self.data_prefix['img'], folder)
            img_ids = sorted(list(map(
                lambda x: int(x.split('.')[0].split('_')[-1]),
                list_dir_or_file(folder_path, recursive=False, list_dir=False, suffix='.jpg', backend_args=self.backend_args)
            )))
            print_log(f'Found {len(img_ids)} in {folder}.', 'current')
            img_ids_list.extend(img_ids)
            for img_id in img_ids:
                self.id2folder[img_id] = folder

        img_ids = img_ids_list
        data_list = []
        for img_id in img_ids:
            if self.custom_structure:
                # tt structure
                data_info = {
                    'img_id': img_id,
                    'img_path': join_path(self.data_prefix['img'],
                                          self.id2folder[img_id],
                                          'img',
                                          self.id2folder[img_id],
                                          f"sa_{img_id}.jpg",
                                          backend_args=self.backend_args),
                    'info_path': join_path(self.data_prefix['img'],
                                           self.id2folder[img_id],
                                           'label',
                                           self.id2folder[img_id],
                                           f"sa_{img_id}.json",
                                           backend_args=self.backend_args),
                }
            else:
                data_info = {
                    'img_id': img_id,
                    'img_path': join_path(self.data_prefix['img'], self.id2folder[img_id], f"sa_{img_id}.jpg", backend_args=self.backend_args),
                    'info_path': join_path(self.data_prefix['img'], self.id2folder[img_id], f"sa_{img_id}.json", backend_args=self.backend_args),
                }
            data_list.append(data_info)
        print_log(f'Found {len(data_list)} in {len(folders)} folders.', 'current')
        return data_list
