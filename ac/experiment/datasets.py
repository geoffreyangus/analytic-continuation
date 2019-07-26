"""
"""
import os
import json

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    Base superclass for all PyTorch dataset implementations.
    """
    def __init__(self, dataset_dir, split=None):
        """
        """
        self.dataset_dir = dataset_dir
        self.split = split

        if split is None:
            sequence_path = os.path.join(dataset_dir, 'sequences.csv')
        else:
            sequence_path = os.path.join(
                dataset_dir, 'split', f'{split}_sequences.csv')
        self.sequence_df = pd.read_csv(sequence_path, index_col=0)

    def __len__(self):
        """
        """
        return len(self.sequence_df)

    def get_sequence(self, sequence_id):
        """
        """
        idx = self.sequence_df.loc[self.sequence_df['sequence_id'] == sequence_id].index[0]
        return self[idx]

    def __getitem__(self, idx):
        """
        """
        pass

    def _apply_transforms(self, images):
        pass


class SequenceDataset(BaseDataset):
    """
    Loads numpy matrices as example sequences.
    """
    def __init__(self, dataset_dir, split=None, task_configs=[]):
        """
        """
        super().__init__(dataset_dir, split)
        self.tasks = [task_config['task'] for task_config in task_configs]

    def __getitem__(self, idx):
        """
        """
        sequence_series = self.sequence_df.iloc[idx]
        sequence_path = sequence_series['sequence_path']
        inputs = self._load_sequence(sequence_path)

        targets = {}
        if len(self.tasks) == 1:
            task = self.tasks[0]
            if task == 'coeff':
                target = [f'{task}_{i}' for i in range(4)]
                targets[task] = torch.tensor(target)
        else:
            for task in self.tasks:
                targets[task] = torch.tensor(sequence_series[task])

        return inputs, targets, {
            'sequence_id': sequence_series.name
        }

    def _load_sequence(self, sequence_path):
        """
        """
        sequence = torch.tensor(np.load(sequence_path)).float()
        return sequence
