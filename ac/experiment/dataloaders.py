"""
"""
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from ac.util import array_like_stack


def mt_collate_fn(batch_list):
    """ Collate function for a multi-task dataset.

    Assumes all inputs are the same size.

    Args:
        batch_list (list) list of sequences
    """
    all_inputs = []
    all_targets = defaultdict(list)
    all_info = []

    for inputs, targets, info in batch_list:
        all_inputs.append(inputs)
        all_info.append(info)
        for task, target in targets.items():
            if len(target.shape) < 1:
                target = target.unsqueeze(dim=0)
            all_targets[task].append(target)

    # stack targets and inputs
    all_targets = {task: array_like_stack(targets)
                   for task, targets in all_targets.items()}
    all_inputs = array_like_stack(all_inputs)
    return all_inputs, all_targets, all_info


class MTDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=6,
                 sampler=None,
                 num_samples=1000,
                 replacement=False,
                 weight_task=None,
                 class_probs=None,
                 pin_memory=False):

        if sampler == 'RandomSampler':
            sampler = RandomSampler(data_source=dataset, num_samples=num_samples,
                                    replacement=True)
            self.num_samples = int(round(num_samples / batch_size))
        else:
            self.num_samples = int(round(len(dataset) / batch_size))

        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, sampler=sampler, pin_memory=pin_memory,
                         collate_fn=mt_collate_fn)

    def __len__(self):
        return self.num_samples
