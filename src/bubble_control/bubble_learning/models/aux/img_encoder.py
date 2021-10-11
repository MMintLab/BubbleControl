import numpy as np
import torch
import torch.nn as nn
import os
import sys

from bubble_control.bubble_learning.models.aux.fc_module import FCModule


class ImageEncoder(nn.Module):
    """
    Module composed by 2D Convolutions followed by FC layers
    """
    def __init__(self, input_size, latent_size, num_convs=3, conv_h_sizes=None, ks=4, num_fcs=2, fc_hidden_size=50, activation='relu'):
        super().__init__()
        self.input_size = input_size # (C_in, W_in, H_in)
        self.latent_size = latent_size
        self.num_convs, self.hidden_dims = self._get_convs_h_sizes(num_convs, conv_h_sizes)
        self.ks = ks
        self.num_fcs = num_fcs
        self.fc_hidden_size = fc_hidden_size
        self.act = self._get_activation(activation) # only used in conv
        self.conv_encoder, self.conv_out_size = self._get_conv_encoder()
        self.fc_encoder = self._get_fc_encoder()

    def forward(self, x):
        batch_size = x.size(0) # shape (Batch_size, ..., C_in, H_in, W_in)
        conv_out = self.conv_encoder(x) # shape (Batch_size, ..., C_out, H_out, W_out)
        fc_in = torch.flatten(conv_out, start_dim=-3) # flatten the last 3 dims
        z = self.fc_encoder(fc_in)
        return z

    def _get_convs_h_sizes(self, num_convs, conv_h_sizes):
        if conv_h_sizes is None:
            hidden_dims = [self.input_size[0]] + [10]*num_convs
        else:
            hidden_dims = [self.input_size[0]] + conv_h_sizes
            num_convs = len(conv_h_sizes)
        return num_convs, hidden_dims

    def _get_conv_encoder(self):
        conv_modules = []
        ks = self.ks
        for i, h_dim in enumerate(self.hidden_dims[:-1]):
            out_dim = self.hidden_dims[i + 1]
            conv_i = nn.Conv2d(in_channels=h_dim, out_channels=out_dim, kernel_size=ks)
            conv_modules.append(conv_i)
            conv_modules.append(self.act)
        conv_encoder = nn.Sequential(*conv_modules)
        conv_img_out_size_wh = self.input_size[1:] + (1 - ks) * self.num_convs
        conv_img_out_size = np.insert(conv_img_out_size_wh, 0, self.hidden_dims[-1])
        return conv_encoder, conv_img_out_size

    def _get_fc_encoder(self):
        fc_in_size = np.prod(self.conv_out_size)
        sizes = [fc_in_size] +[self.fc_hidden_size]*(self.num_fcs-1) + [self.latent_size]
        fc_encoder = FCModule(sizes, activation='relu')
        return fc_encoder

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise NotImplementedError('Activation {} not supported'.format(activation))