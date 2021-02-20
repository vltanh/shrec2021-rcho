import torch.nn as nn

from torchan.losses.classification.crossentropy import CrossEntropyLoss

__all__ = ['MultiClassifierLoss']


class MultiClassifierLoss(CrossEntropyLoss):
    def forward(self, output, target):
        # output: B, R+1, C
        # target: B
        B, R, C = output.size()

        target = target.unsqueeze(0).repeat(R, 1).T  # R+1, B
        target = target.reshape(-1)  # (R+1)*B

        output = output.reshape(-1, C)  # B*(R+1), C

        return super().forward(output, target)
