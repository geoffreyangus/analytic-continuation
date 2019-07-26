"""
"""
import torch
import torch.nn as nn
from torch.nn import MSELoss, KLDivLoss

from ac.util import get_batch_size


class MTKLDivLoss(KLDivLoss):
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
        return loss

    def _compute_loss(self, inputs, targets):
        """
        """
        losses = {}
        for task in targets.keys():
            Y_hat = inputs[task]
            losses[task] = super().forward(Y_hat, targets[task])
        return losses

    def _balance(self, losses, targets):
        """
        """
        for task in targets.keys():
            if self.reduction == "mean":
                losses[task] = losses[task].mean()
            if self.reduction == "sum":
                losses[task] = losses[task].sum()

        loss = sum(losses.values())
        return loss


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
        return loss

    def _compute_loss(self, inputs, targets):
        """
        """
        losses = {}
        for task in targets.keys():
            Y_hat = inputs[task]
            losses[task] = super().forward(Y_hat, targets[task])
        return losses

    def _balance(self, losses, targets):
        """
        """
        for task in targets.keys():
            if self.reduction == "mean":
                losses[task] = losses[task].mean()
            if self.reduction == "sum":
                losses[task] = losses[task].sum()

        loss = sum(losses.values())
        return loss


class MTSoftCrossEntropyLoss(nn.Module):
    """Computes the CrossEntropyLoss while accepting probabilistic (float)
    targets in a multi-task setting. This is a modified implementation of
    SoftCrossEntropyLoss as presented by the HazyResearch group at Stanford.

    Source: https://github.com/HazyResearch/metal/

    Args:
        weight: a tensor of relative weights to assign to each class.
            the kwarg name 'weight' is used to match CrossEntropyLoss
        reduction: how to combine the elementwise losses
            'none': return an unreduced list of elementwise losses
            'mean': return the mean loss per elements
            'sum': return the sum of the elementwise losses
    Accepts:
        input: An [n, k] float tensor of prediction logits (not probabilities)
        target: An [n, k] float tensor of target probabilities
    """

    def __init__(self, weight=None, reduction="mean"):
        super().__init__()
        # Register as buffer is standard way to make sure gets moved /
        # converted with the Module, without making it a Parameter
        if weight is None:
            self.weight = None
        else:
            # Note: Sets the attribute self.weight as well
            self.register_buffer("weight", torch.FloatTensor(weight))
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        """
        losses = self._compute_loss(inputs, targets)
        loss = self._balance(losses, targets)
        return loss

    def _compute_loss(self, inputs, targets):
        """
        """
        losses = {}
        for task in targets.keys():
            n, k = inputs[task].shape
            # Note that t.new_zeros, t.new_full put tensor on same device as t
            cum_losses = inputs[task].new_zeros(n)
            for y in range(k):
                cls_idx = inputs[task].new_full((n,), y, dtype=torch.long)
                y_loss = F.cross_entropy(inputs[task], cls_idx, reduction="none")
                if self.weight is not None:
                    y_loss = y_loss * self.weight[y]
                cum_losses += target[task][:, y].float() * y_loss
            losses[task] = cum_losses
        return losses

    def _balance(self, losses, targets):
        """
        """
        for task in targets.keys():
            if self.reduction == "none":
                continue
            elif self.reduction == "mean":
                losses[task] = losses[task].mean()
            elif self.reduction == "sum":
                losses[task] = losses[task].sum()
            else:
                raise ValueError(f"Unrecognized reduction: {self.reduction}")

        loss = sum(losses.values())
        return loss

