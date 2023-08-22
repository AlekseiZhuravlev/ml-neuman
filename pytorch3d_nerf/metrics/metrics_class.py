import torch

class MetricsCalculator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics = {}