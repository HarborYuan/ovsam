from typing import Union, Dict, Tuple, List
from collections import defaultdict

import mmengine
import torch
from mmdet.models import BaseDetector
from mmdet.models.detectors.base import ForwardResults
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmengine.optim import OptimWrapper
from mmengine.structures import InstanceData
from torch import Tensor

import torch.nn as nn

COCO_CAT2LABEL = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14,
    17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27,
    33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40,
    47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53,
    60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66,
    77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}


@MODELS.register_module()
class JsonLoader(BaseDetector):

    def __init__(self, json_path: str, dataset='coco'):
        super().__init__(init_cfg=None)
        json = mmengine.load(json_path)
        data_dict = defaultdict(list)
        for item in json:
            image_id = item['image_id']
            data_dict[image_id].append(item)
        instance_data = dict()
        for image_id in data_dict:
            if dataset == 'coco':
                labels = torch.tensor([COCO_CAT2LABEL[itm['category_id']] for itm in data_dict[image_id]])
            elif dataset == 'lvis':
                labels = torch.tensor([itm['category_id'] - 1 for itm in data_dict[image_id]])
            elif dataset == 'trainid':
                labels = torch.tensor([itm['category_id'] for itm in data_dict[image_id]])

            scores = torch.tensor([itm['score'] for itm in data_dict[image_id]])
            bboxes = []
            for itm in data_dict[image_id]:
                x, y, h, w = itm['bbox']
                bboxes.append([x, y, x + h, y + w])
            bboxes = torch.tensor(bboxes)
            instance_data[image_id] = InstanceData(
                labels=labels,
                scores=scores,
                bboxes=bboxes
            )
        self.instance_data = instance_data
        self.dummy_param = nn.Parameter(torch.zeros((1,)))

    def extract_feat(self, batch_inputs: Tensor):
        pass

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            # data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        # data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        # data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList = None) -> Tuple[Tensor]:
        raise NotImplementedError

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[Dict, List]:
        raise NotImplementedError

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> Union[Dict, List]:
        for data_samples in batch_data_samples:
            img_id = data_samples.img_id
            if img_id in self.instance_data:
                pred_instances = self.instance_data[img_id]
            else:
                pred_instances = InstanceData(
                    labels=torch.zeros((0,), dtype=torch.long),
                    scores=torch.zeros((0,), dtype=torch.float32),
                    bboxes=torch.zeros((0, 4), dtype=torch.float32),
                )
                print(f"Warning, {img_id} is not in the json file.")
            data_samples.pred_instances = pred_instances
        return batch_data_samples
