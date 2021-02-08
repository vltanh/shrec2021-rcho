import torch.nn as nn
from torch.nn import functional as F

from utils import getter

__all__ = ['ImageMaskExtractor']

class Extractor(nn.Module):
    def freeze(self):
        for p in self.extractor.parameters():
            p.requires_grad = False

class ImageExtractor(Extractor):
    def get_embedding(self, x):
        x = self.get_feature_map(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x
        
class ImageMaskExtractor(ImageExtractor):
    def __init__(self, ext_cfg):
        super().__init__()
        self.ext = getter.get_instance(ext_cfg)
        self.feature_dim = self.ext.feature_dim

    def get_feature_map(self, x):
        im = x[:, :3]
        mask = x[:, 3]

        ft_map = self.ext.get_feature_map(im)

        mask = mask.unsqueeze(1).float()
        mask = F.interpolate(mask, size=ft_map.shape[-2:],
                             mode='bilinear', align_corners=True)

        ft_map *= mask
        ft_map = F.adaptive_avg_pool2d(ft_map, (1, 1))
        
        return ft_map