from mmdet.models import DetDataPreprocessor
from mmdet.registry import MODELS


@MODELS.register_module()
class DetDataInferenceTimePreprocessor(DetDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> dict:
        data = super().forward(data=data, training=True)
        return data
