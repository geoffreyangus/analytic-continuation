"""
"""
import torch
from torch.nn import MSELoss, KLDivLoss

from ac.util import get_batch_size


class MTMSELoss(MSELoss):
    """
    """
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        """
        """
        super().__init__(size_average, reduce, reduction)

    def forward(self, inputs, targets):
        """
        """
        losses = self._compute_loss(inputs, targets)
        loss = self._balance(losses, targets)
        # print('--')
        # print(f'final loss:\t{loss}')
        return loss

    def _compute_loss(self, inputs, targets):
        """
        """
        # print('--')
        # print('MTMSELoss')
        losses = {}
        for task, Y in targets.items():
            Y_hat = inputs[task]
            # print(f"task: {task}")
            # print(f'- true:\t{Y.data}')
            # print(f'- pred:\t{Y_hat.data}')
            losses[task] = super().forward(Y_hat, Y)
            # print(f"- loss:\t{losses[task]}")
        return losses

    def _balance(self, losses, targets):
        """
        """
        # print('--')
        for task, Y in targets.items():
            # broadcast multiply task-specific coefficients (uses all targets)
            if self.reduction == "mean":
                losses[task] = losses[task].mean()
            if self.reduction == "sum":
                losses[task] = losses[task].sum()
            # print(f'task: {task}')
            # print(f'- loss:\t{losses[task]}')

        loss = sum(losses.values())
        return loss