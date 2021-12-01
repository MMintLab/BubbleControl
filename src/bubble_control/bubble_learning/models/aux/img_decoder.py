import numpy as np
import torch
import torch.nn as nn
import os
import sys

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/ContactInvariances')[0], 'ContactInvariances')
package_path = os.path.join(project_path, 'contact_invariances', 'learning')
sys.path.append(project_path)

from bubble_control.bubble_learning.models.aux.fc_module import FCModule
from bubble_control.bubble_learning.models.aux.img_encoder import ImageEncoder


class ImageDecoder(nn.Module):
    """
    Module composed by FC layers and 2D Inverse Convolutions (Transposed Conv)
    """
    def __init__(self, output_size, latent_size, num_convs=3, conv_h_sizes=None, ks=4, stride=1, num_fcs=2, fc_hidden_size=50, activation='relu'):
        super().__init__()
        self.output_size = output_size # (C_out, W_out, H_out)
        self.latent_size = latent_size
        self.num_convs, self.hidden_dims = self._get_convs_h_sizes(num_convs, conv_h_sizes)
        self.ks = ks
        self.stride = stride
        self.num_fcs = num_fcs
        self.fc_hidden_size = fc_hidden_size
        self.act = self._get_activation(activation) # only used in conv
        self.conv_decoder, self.conv_in_size = self._get_conv_decoder()
        self.fc_decoder = self._get_fc_decoder()

    def forward(self, z):
        batch_size = z.size(0) # shape (Batch_size, ..., latent_size)
        conv_in = self.fc_decoder(z) # adjust the shape for the convolutions
        conv_in = conv_in.view(z.size()[:-1] + tuple(self.conv_in_size))  # shape (Batch_size, ..., C_in, H_in, W_in)
        conv_out = self.conv_decoder(conv_in) # shape (Batch_size, ..., C_out, H_out, W_out)
        return conv_out

    def _get_convs_h_sizes(self, num_convs, conv_h_sizes):
        if conv_h_sizes is None:
            hidden_dims = [self.output_size[0]]*num_convs + [self.output_size[0]]
        else:
            hidden_dims = conv_h_sizes + [self.output_size[0]]
            num_convs = len(conv_h_sizes)
        return num_convs, hidden_dims

    def _get_conv_decoder(self):
        conv_modules = []
        ks = self.ks
        stride = self.stride
        for i, h_dim in enumerate(self.hidden_dims[:-1]):
            out_dim = self.hidden_dims[i + 1]
            conv_i = nn.ConvTranspose2d(in_channels=h_dim, out_channels=out_dim, kernel_size=ks, stride=stride)
            conv_modules.append(conv_i)
            if i < len(self.hidden_dims)-2:
                conv_modules.append(self.act)
        if len(conv_modules) > 0:
            conv_encoder = nn.Sequential(*conv_modules)
        else:
            conv_encoder = nn.Identity() # no operation needed since there are no convolutions
        # compute the tensor sizes:

        conv_img_in_size_wh = self.output_size[1:]
        for i in range(self.num_convs):
            conv_img_in_size_wh = (conv_img_in_size_wh - ks)/stride + 1
        conv_img_in_size = np.insert(conv_img_in_size_wh, 0, self.hidden_dims[0]) # ( C_in, H_in, W_in)
        return conv_encoder, conv_img_in_size

    def _get_fc_decoder(self):
        fc_out_size = int(np.prod(self.conv_in_size))
        sizes = [self.latent_size] + [self.fc_hidden_size]*(self.num_fcs-1) + [fc_out_size]
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