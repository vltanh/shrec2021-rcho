import torch.nn as nn

from .extractor_network import ImageExtractor
from efficientnet_pytorch import EfficientNet


class EfficientNetExtractor(ImageExtractor):
    def __init__(self, version):
        super().__init__()
        assert version in range(9)
        self.extractor = EfficientNet.from_pretrained(
            f'efficientnet-b{version}')
        self.feature_dim = self.extractor._fc.in_features

    def get_feature_map(self, x):
        return self.extractor.extract_features(x)