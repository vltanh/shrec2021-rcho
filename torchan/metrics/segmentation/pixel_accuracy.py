import torch

__all__ = ['PixelAccuracy']


class PixelAccuracy():
    def __init__(self, nclasses, ignore_index=None):
        super().__init__()
        assert nclasses > 0

        self.nclasses = nclasses
        self.ignore_index = ignore_index
        self.reset()

    def update(self, output, target):
        prediction = torch.argmax(output, dim=1)

        image_size = target.size(1) * target.size(2)

        ignore_mask = torch.zeros(target.size()).bool().to(target.device)
        if self.ignore_index is not None:
            ignore_mask = (target == self.ignore_index).bool()
        ignore_size = ignore_mask.sum((1, 2))

        correct = ((prediction == target) | ignore_mask).sum((1, 2))
        acc = (correct - ignore_size + 1e-6) / \
            (image_size - ignore_size + 1e-6)

        acc = acc.cpu()
        self.total_correct += acc.sum(0)
        self.sample_size += acc.size(0)

    def value(self):
        return (self.total_correct / self.sample_size).item()

    def reset(self):
        self.total_correct = 0
        self.sample_size = 0

    def summary(self):
        print(f'Pixel Accuracy: {self.value():.6f}')
