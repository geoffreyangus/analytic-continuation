"""
"""
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    """
    def __init__(self, dataset_dir, split=None):
        """
        """
        self.dataset_dir = dataset_dir
        self.split = split

        if split is None:
            exams_path = os.path.join(dataset_dir, 'series.csv')
        else:
            exams_path = os.path.join(
                dataset_dir, 'split', f'{split}_series.csv')
        self.exams_df = pd.read_csv(exams_path)

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
        return len(self.exams_df)

    def get_exam(self, exam_id):
        """
        """
        idx = self.exams_df.loc[self.exams_df['exam_id'] == exam_id].index[0]
        return self[idx]

    def __getitem__(self, idx):
        """
        """
        pass

    def _apply_transforms(self, images):
        pass
