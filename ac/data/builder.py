"""
Implements (typically) one-use dataset builders that compile raw .dat files.
"""
import os
import re
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import numpy as np

from ac.util import Process, ensure_dir_exists


class Builder(Process):
    """
    """
    def __init__(self, dir, data_dir, output_dir,
                 num_materials=4, max_timesteps=None):
        """
        """
        super().__init__(dir)
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_materials = num_materials
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
        sequences = os.listdir(self.data_dir)
        for filename in tqdm(sequences, total=len(sequences)):
            if os.path.splitext(filename)[1] != '.dat':
                continue
            ids = list(re.finditer(r'[0-9]+', filename))
            beta_id = ids[0][0]
            sequence_id = ids[1][0]

            if 'beta' not in data_dict[sequence_id]:
                data_dict[sequence_id]['beta'] = {}

            coeffs, parameters, data = self._read_dat_file(
                os.path.join(self.data_dir, filename)
            )
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
        for sequence_id, data in tqdm(data_dict.items(), total=len(data_dict)):
            output = {'sequence_id': sequence_id}

            for coeff_idx, coeff in enumerate(data['coeffs']):
                output[f'coeff_{coeff_idx}'] = coeff
            for param_idx, param in enumerate(data['parameters']):
                output[f'param_{param_idx}'] = param

            beta_dict = data_dict[sequence_id]['beta']
            # assumes all sequences have the same beta_vals
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
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if i == 0:
                    coeffs, i = self._extract_vals(lines, i)
                    parameters, i = self._extract_vals(lines, i)
                else:
                    data.append(np.fromstring(lines[i], sep='\t'))
                    i += 1
        data = np.array(data)[:,1:]
        return coeffs, parameters, data

    def _extract_vals(self, lines, i):
        """
        """
        line = ''
        offset = 0
        while True:
            line += lines[i + offset]
            line = line.replace('\n', '')
            parameters_str = line.split('\t')
            parameters_str = parameters_str[1:]
            parameters = []
            for c in parameters_str:
                val = -1.0
                if 'N/A' not in c:
                    try:
                        val = float(c)
                    except:
                        val = 0.0
                parameters.append(val)
            offset += 1
            if len(parameters) == self.num_materials:
                break
        i += offset
        return parameters, i
