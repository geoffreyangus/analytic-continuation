"""
"""
import logging

import torch
import torch.nn as nn
import torch.optim as optims
import torch.optim.lr_scheduler as schedulers
from tqdm import tqdm

from ac.models.base import BaseModel
from ac.analysis.metrics import Metrics
import ac.models.modules.encoders as encoders
import ac.models.modules.decoders as decoders
import ac.models.modules.losses as losses
from ac.util import place_on_gpu, get_batch_size

class RegressionModel(BaseModel):
    """
    """

    def __init__(self, cuda=False, devices=[0], pretrained_configs=[],
                 encoder_config=None, decoder_config=None,
                 loss_config=None, optim_config="Adam", scheduler_config=None,
                 task_configs=[]):
        """
        """
        super().__init__(cuda, devices)

        encoder_class = encoder_config["class"]
        encoder_args = encoder_config["args"]
        self.encoder = getattr(encoders, encoder_class)(**encoder_args)
        self.encoder = nn.DataParallel(self.encoder, device_ids=self.devices)

        decoder_class = decoder_config["class"]
        decoder_args = decoder_config["args"]
        self.decoder = getattr(decoders, decoder_class)(task_configs, **decoder_args)

        loss_class = loss_config["class"]
        loss_args = loss_config["args"]
        self.loss_fn = getattr(losses, loss_class)(**loss_args)

        self._post_init(optim_config, scheduler_config, pretrained_configs)

    def forward(self, inputs, targets):
        """
        Args:
            inputs  (torch.Tensor) a (batch_size, ...) shaped input tensor
            targets (dict) a dictionary mapping task names to targets.
        Return:
            outputs (dict) a dict that matches task names to targets.
        """
        encoding = self.encoder(inputs)
        outputs = self.decoder(encoding)
        assert(outputs.keys() == targets.keys())
        return outputs

    def calculate_loss(self, inputs, targets):
        """
        """
        return self.loss_fn(inputs=inputs, targets=targets)

    def predict(self, inputs):
        """
        """
        encoding = self._encode(inputs)
        probs = self._predict_tasks(encoding)
        return probs

    def _encode(self, inputs):
        """Accepts time series data and outputs an encoding"""
        encoding = self.encoder(inputs)
        return encoding

    def _predict_tasks(self, encoding):
        """
        """
        outputs = self.decoder(encoding)
        # probs = {task: out / out.sum(dim=1).unsqueeze(1)
        #          for task, out in outputs.items()}
        return outputs


class CoeffsRegressionModel(RegressionModel):
    """
    """
    def __init__(self, cuda=False, devices=[0], pretrained_configs=[],
                 encoder_config=None, decoder_config=None,
                 loss_config=None, optim_config="Adam", scheduler_config=None,
                 task_configs=[]):
        """
        """
        super().__init__(cuda, devices, pretrained_configs,
                         encoder_config, decoder_config,
                         loss_config, optim_config, scheduler_config,
                         task_configs)

        self.coeff_tasks = ['coeff_0',
                            'coeff_1',
                            'coeff_2',
                            'coeff_3']

    def _predict_tasks(self, encoding):
        """
        """
        outputs = self.decoder(encoding)
        tasks = outputs.keys()
        assert set(tasks).issubset(self.coeff_tasks), 'Coeffs must be tasks.'

        batch_size = get_batch_size(outputs['coeff_0'])
        for i in range(batch_size):
            sequence = []
            for coeff_task_idx, coeff_task in enumerate(self.coeff_tasks):
                sequence.append(outputs[coeff_task][i])
            sequence = torch.tensor(sequence)
            sequence = sequence + torch.min(sequence) # non-negative
            sequence = sequence / torch.sum(sequence)
            for coeff_task_idx, coeff_task in enumerate(self.coeff_tasks):
                outputs[coeff_task][i] = sequence[coeff_task_idx]
        return outputs
