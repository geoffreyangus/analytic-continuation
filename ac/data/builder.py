"""
Implements (typically) one-use dataset builders that compile raw .dat files.
"""
import os
import re
from collections import defaultdict

import pandas as pd
import numpy as np

from ac.util import Process, ensure_dir_exists


class Builder(Process):
    """
    """
    def __init__(self, dir, data_dir, output_dir, max_timesteps=None):
        """
        """
        super().__init__(dir)
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_timesteps = max_timesteps

    def _run(self, overwrite=False):
        """
        """
        ensure_dir_exists(self.output_dir)
        data_dict = self._load_data()
        self._save_data(data_dict)

    def _load_data(self):
        """
        """
        data_dict = defaultdict(dict)
        for entry in os.scandir(self.data_dir):
            ids = list(re.finditer(r'[0-9]+', entry.name))
            beta_id = ids[0][0]
            sequence_id = ids[1][0]

            if 'beta' not in data_dict[sequence_id]:
                data_dict[sequence_id]['beta'] = {}

            coeffs, parameters, data = self._read_dat_file(entry.path)
            data_dict[sequence_id]['beta'][beta_id] = data

            if 'coeffs' in data_dict[sequence_id]:
                assert coeffs == data_dict[sequence_id]['coeffs'], (
                    f"coeffs {coeffs} != {data_dict[sequence_id]['coeffs']}"
                )
            else:
                data_dict[sequence_id]['coeffs'] = coeffs

            if 'parameters' in data_dict[sequence_id]:
                assert parameters == data_dict[sequence_id]['parameters'], (
                    f"params {parameters} != {data_dict[sequence_id]['parameters']}"
                )
            else:
                data_dict[sequence_id]['parameters'] = parameters
        return data_dict

    def _save_data(self, data_dict):
        """
        """
        beta_vals = None
        output_df = []
        for sequence_id, data in data_dict.items():
            output = {'sequence_id': sequence_id}

            for coeff_idx, coeff in enumerate(data['coeffs']):
                output[f'coeff_{coeff_idx}'] = coeff
            for param_idx, param in enumerate(data['parameters']):
                output[f'param_{param_idx}'] = param

            beta_dict = data_dict[sequence_id]['beta']
            if not beta_vals:
                beta_vals = sorted(beta_dict.keys(),
                                   key=lambda x: int(x))
            X_list = []
            for beta in beta_vals:
                x_i = beta_dict[beta]
                X_list.append(x_i)
            X = np.zeros((len(beta_vals), max([len(x_i) for x_i in X_list])))
            for i, x_i in enumerate(X_list):
                X[i,:len(x_i)] = x_i.flat

            output_path = os.path.join(self.output_dir, f'sequence_{sequence_id}')
            np.save(output_path, X)
            output['sequence_path'] = f'{output_path}.npy'
            output_df.append(output)

        output_df = pd.DataFrame(output_df)
        output_df = output_df.set_index('sequence_id')
        output_df.index = output_df.index.astype(int)
        output_df = output_df.sort_index()
        output_df.to_csv(os.path.join(self.output_dir, 'sequences.csv'),
                         index_label='sequence_id')
        output_df.to_csv(os.path.join(self.dir, 'sequences.csv'),
                         index_label='sequence_id')

    def _read_dat_file(self, filepath):
        """
        """
        data = []
        with open(filepath, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    coeffs = line.split('\t')
                    coeffs = coeffs[1:]
                    coeffs = [float(c) if c != 'N/A' else -1.0
                              for c in coeffs]
                elif i == 1:
                    parameters = line.split('\t')
                    parameters = parameters[1:]
                    parameters = [float(p) if 'N/A' not in p else np.nan
                                  for p in parameters]
                else:
                    data.append(np.fromstring(line, sep='\t'))
        data = np.array(data)[:,1:]
        return coeffs, parameters, data
