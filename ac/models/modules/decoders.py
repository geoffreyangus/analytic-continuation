"""
"""
import torch
import torch.nn as nn


class MTDecoder(nn.Module):
    """
    """
    def __init__(self, task_configs):
        """
        """
        super().__init__()

        self.task_heads = nn.ModuleDict()
        for task_config in task_configs:
            task = task_config['task']
            task_head_config = task_config['task_head_config']
            task_head_class = task_head_config['class']
            task_head_args = task_head_config['args']
            self.task_heads[task] = globals()[task_head_class](**task_head_args)

    def forward(self, encoding):
        """
        """
        preds = {}
        for task, task_head in self.task_heads.items():
            preds[task] = task_head(encoding)
        return preds


class LinearDecoder(nn.Module):
    """
    """

    def __init__(self, input_size=256, num_classes=4):
        """
        """
        super().__init__()
        self.network = nn.Linear(input_size, num_classes)

    def forward(self, encoding):
        """
        """
        return self.network(encoding)
