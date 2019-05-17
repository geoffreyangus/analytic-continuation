"""
"""
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
        self.sequence_df = pd.read_csv(sequence_path)

        self._get_data_dir()

    def _get_data_dir(self):
        """
        """
        # load dataset name and directory from dataset params file
        with open(os.path.join(self.dataset_dir, "params.json")) as f:
            params = json.load(f)
            self.dataset_name = params["dataset_name"]
            self.data_dir = params["data_dir"]

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

    def __init__(self, dataset_dir, split=None):
        """
        """
        super().__init__(dataset_dir, split)

    def __getitem__(self, idx):
        """
        """
        pass

