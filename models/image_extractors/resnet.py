from torch import nn
from torchvision import models


from .extractor_network import ImageExtractor


class ResNetExtractor(ImageExtractor):
    arch = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }

    def __init__(self, version, use_pretrained, is_frozen=False):
        super().__init__()
        assert version in ResNetExtractor.arch, \
            f'{version} is not implemented.'
        cnn = ResNetExtractor.arch[version](pretrained=use_pretrained)
        self.extractor = nn.Sequential(*list(cnn.children())[:-2])
        self.feature_dim = cnn.fc.in_features
        if is_frozen:
            self.freeze()

    def get_feature_map(self, x):
        return self.extractor(x)
