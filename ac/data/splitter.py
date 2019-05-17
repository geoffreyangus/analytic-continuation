"""
Implements (typically) one-use dataset splitters.
"""
import os

import pandas as pd
import numpy as np

from ac.util import Process


class Splitter(Process):
    """
    """
    def __init__(self, dir, seed=123, aggregate_csv_path="", split_totals={}):
        """
        Splits the aggregate_csv by the numbers given in split_totals.

        Exactly one of the split_totals can be -1. This split will take the
        rest of the examples after splitting. NOTE: assumes that the index
        of the aggregate_csv is a unique identifier.
        """
        super().__init__(dir)

        self.seed = seed
        np.random.seed(self.seed)

        self.split_totals = split_totals
        assert list(split_totals.values()).count(-1) == 1, (
            'Exactly on the split_total entries can be -1.'
        )

        self.aggregate_df = pd.read_csv(aggregate_csv_path, index_col=0)
        assert sum(split_totals.values()) + 1 < len(self.aggregate_df), (
            f'Not enough values in csv to split into {split_totals.values()}'
        )

    def _run(self, overwrite=False):
        """
        """
        split_to_df = self._split()
        self._verify(split_to_df)
        self._write(split_to_df)

    def _split(self):
        """
        Implements a random split by shuffling the original aggregate csv.
        """
        m = len(self.aggregate_df)
        idxs = np.random.choice(self.aggregate_df.index.tolist(), m,
                                replace=False)
        split_to_df = {}
        final_split = ''
        offset = 0
        for split, totals in self.split_totals.items():
            if totals == -1:
                final_split = split
                continue
            split_to_df[split] = self.aggregate_df.loc[idxs[offset: offset + totals]]
            offset = offset + totals
        split_to_df[final_split] = self.aggregate_df.loc[idxs[offset:]]
        return split_to_df

    def _verify(self, split_to_df):
        """
        Verifies that the splits are correct.
        """
        index_set = set(self.aggregate_df.index)

        split_indices = set()
        for split, df in split_to_df.items():
            split_indices.update(df.index)

        assert len(index_set) <= len(split_indices), (
            f'Examples {index_set - split_indices}' + 'were inadvertently lost.'
        )
        assert len(index_set) >= len(split_indices), (
            f'Examples {split_indices - index_set}' + 'were inadvertently added.'
        )

    def _write(self, split_to_df):
        """
        Saves all split dataframes.
        """
        for split, df in split_to_df.items():
            df.to_csv(os.path.join(self.dir, f'{split}_sequences.csv'),
                    index_label='sequence_id')
