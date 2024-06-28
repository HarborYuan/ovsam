import copy
from typing import Union, Tuple, Dict, List, Optional

import torch
from mmdet.models.detectors.base import ForwardResults
from mmdet.utils import ConfigType
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import SampleList, OptSampleList


@MODELS.register_module()
class DetSeg(BaseModel):
    def __init__(self,
                 det_model: ConfigType,
                 seg_model: Optional[ConfigType] = None,
                 cls_model: Optional[ConfigType] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        # The data_preprocessor should be None
        assert data_preprocessor is None

        self.det_model = MODELS.build(det_model)
        if seg_model is not None:
            self.seg_model = MODELS.build(seg_model)
        else:
            self.seg_model = None
        if cls_model is not None:
            self.cls_model = MODELS.build(cls_model)
        else:
            self.cls_model = None

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
        det_results = {'inputs': batch_inputs, 'data_samples': batch_data_samples}
        det_results = self.det_model.test_step(det_results)

        if self.seg_model is not None:
            seg_results = {'inputs': batch_inputs, 'data_samples': det_results}
            seg_results = self.seg_model.test_step(seg_results)
        else:
            seg_results = det_results

        if self.cls_model is not None:
            _seg_results = copy.deepcopy(seg_results)
            cls_results = {'inputs': batch_inputs, 'data_samples': _seg_results}
            cls_results = self.cls_model.test_step(cls_results)
            for idx, data_samples in enumerate(seg_results):
                data_samples.pred_instances.labels = cls_results[idx].pred_instances.labels
        return seg_results
