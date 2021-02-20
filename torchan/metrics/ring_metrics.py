from torchan.utils import getter

__all__ = ['RingMetric']


def take_last(x):
    return x[:, -1]


class RingMetric:
    def __init__(self, metric_cfg, strategy='last'):
        self.metric = getter.get_instance(metric_cfg)
        self.strategy = {
            'last': take_last
        }[strategy]

        self.reset = self.metric.reset
        self.value = self.metric.value
        self.summary = self.metric.summary

    def update(self, output, target):
        return self.metric.update(self.strategy(output), target)
