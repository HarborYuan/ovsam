from typing import List

from mmengine import get_local_path, list_from_file, join_path, scandir, print_log

from mmdet.datasets import BaseDetDataset


class SAMDataset(BaseDetDataset):

    def __init__(self, *args, img_map_suffix='.jpg', **kwargs):
        self.img_map_suffix = img_map_suffix
        self.id2folder = dict()
        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        print_log('Starting to load sam dataset', 'current')
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            folders = list_from_file(local_path)

        img_ids_list = []
        for folder in folders:
            folder_path = join_path(self.data_prefix['img'], folder)
            img_ids = sorted(list(map(
                lambda x: int(x.split('.')[0].split('_')[1]),
                scandir(folder_path, recursive=False, suffix='.jpg')
            )))
            img_ids_list.extend(img_ids)
            for img_id in img_ids:
                self.id2folder[img_id] = folder

        img_ids = img_ids_list
        data_list = []
        for img_id in img_ids:
            data_info = {
                'img_id': img_id,
                'img_path': join_path(self.data_prefix['img'], self.id2folder[img_id], f"sa_{img_id}.jpg"),
                'info_path': join_path(self.data_prefix['img'], self.id2folder[img_id], f"sa_{img_id}.json"),
            }
            data_list.append(data_info)
        print_log(f'Found {len(data_list)} in {len(folders)} folders.', 'current')
        return data_list
