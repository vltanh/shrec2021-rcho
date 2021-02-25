import torch
import torch.nn as nn
import torch.nn.functional as F

from torchan.utils import getter

__all__ = ['BaseRingExtractor',
           'Base3DObjectRingsExtractor',
           'MultiClassifier3DObjectRingsExtractor']


class BaseRingExtractor(nn.Module):
    def __init__(self, extractor_cfg, hidden_dim):
        super().__init__()
        self.cnn = getter.get_instance(extractor_cfg)
        self.cnn_feature_dim = self.cnn.feature_dim  # D
        self.feature_dim = 2 * hidden_dim  # D'
        self.lstm = nn.LSTM(self.cnn_feature_dim,
                            hidden_dim,
                            batch_first=True,
                            bidirectional=True)

    def get_embedding(self, x):
        # x: [B, V, C, H, W]
        B, V, C, H, W = x.size()
        x = x.reshape(B*V, C, H, W)  # B*V, C, H, W
        x = self.cnn.get_embedding(x)  # B*V, D
        x = x.reshape(B, V, self.cnn_feature_dim)  # B, V, D
        x, _ = self.lstm(x)  # B, V, D'
        x = x.mean(1)  # B, D'
        return x


class Base3DObjectRingsExtractor(nn.Module):
    def __init__(self, nrings, ring_ext_cfg, nheads, dropout=0.0, reverse=False):
        super().__init__()
        self.reverse = reverse
        if reverse:
            nrings = 12
        self.ring_exts = nn.ModuleList([
            getter.get_instance(ring_ext_cfg)
            for _ in range(nrings)
        ])
        self.view_feature_dim = self.ring_exts[0].feature_dim  # D
        self.feature_dim = self.view_feature_dim  # D'
        self.attn = nn.MultiheadAttention(self.feature_dim, nheads, dropout)

    def forward(self, x):
        # x: B, R, V, C, H, W
        if self.reverse:
            x = x.transpose(1, 2)
        x = torch.cat([
            ring_ext.get_embedding(x[:, i]).unsqueeze(1)
            for i, ring_ext in enumerate(self.ring_exts)
        ], dim=1)  # B, R, D
        x = x.transpose(0, 1)  # R, B, D
        x, _ = self.attn(x, x, x)  # R, B, D
        x = x.mean(0)  # B, D
        return x

    def get_embedding(self, x):
        # x: B, R, V, C, H, W
        return self.forward(x)


class MultiClassifier3DObjectRingsExtractor(Base3DObjectRingsExtractor):
    def forward(self, x):
        # x: B, R, V, C, H, W
        if self.reverse:
            x = x.transpose(1, 2)
        x = torch.cat([
            ring_ext.get_embedding(x[:, i]).unsqueeze(1)  # B, 1, D
            for i, ring_ext in enumerate(self.ring_exts)
        ], dim=1)  # B, R, D
        x = x.transpose(0, 1)  # R, B, D
        emb, _ = self.attn(x, x, x)  # R, B, D
        emb = emb.mean(0, keepdim=True)  # 1, B, D
        x = torch.cat([x, emb], dim=0)  # R+1, B, D
        x = x.transpose(0, 1)  # B, R+1, D
        return x

    def get_embedding(self, x):
        # x: B, R, V, C, H, W
        return self.forward(x)[:, -1]
