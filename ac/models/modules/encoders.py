"""
"""
import torch
import torch.nn as nn


class SimpleEncoder(nn.Module):
    """
    One-dimensional convolutional network encoder.
    """
    def __init__(self, conv_layer_configs=[], linear_layer_configs=[]):
        """
        """
        super().__init__()

        conv_layers = nn.ModuleList()
        for layer_config in conv_layer_configs:
            layer_class = layer_config['class']
            layer_args = layer_config['args']
            layer = getattr(nn, layer_class)(**layer_args)
            conv_layers.append(layer)
        self.conv_layers = nn.Sequential(*conv_layers)

        linear_layers = nn.ModuleList()
        for layer_config in linear_layer_configs:
            layer_class = layer_config['class']
            layer_args = layer_config['args']
            layer = getattr(nn, layer_class)(**layer_args)
            linear_layers.append(layer)
        self.linear_layers = nn.Sequential(*linear_layers)

    def forward(self, inputs):
        """
        """
        encoding = self.conv_layers(inputs)
        encoding = encoding.view(inputs.shape[0], -1)
        encoding = self.linear_layers(encoding)
        return encoding
