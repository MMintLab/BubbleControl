import pdb

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import numpy as np
import os
import sys

from bubble_control.bubble_learning.models.aux.fc_module import FCModule
from bubble_control.bubble_learning.models.aux.img_encoder import ImageEncoder
from bubble_control.bubble_learning.models.aux.img_decoder import ImageDecoder


class BubbleDynamicsResidualModel(pl.LightningModule):
    """
    Model designed to model the bubbles dynamics.
    Given s_t, and a_t, it produces ∆s, where s_{t+1} = s_t + ∆s
     * Here s_t is composed by:
        - Depth image from each of the bubbles
        - Wrench information
        - End effector pose: Position and orientation of the end effector
    * The depth images are embedded into a vector which is later concatenated with the wrench and pose information
    """
    def __init__(self, input_sizes, img_embedding_size, encoder_num_convs=3, decoder_num_convs=3, encoder_conv_hidden_sizes=None, decoder_conv_hidden_sizes=None, ks=3, num_fcs=2, num_encoder_fcs=2, num_decoder_fcs=2, fc_h_dim=100, skip_layers=None, lr=1e-4, dataset_params=None, activation='relu'):
        super().__init__()
        self.input_sizes = input_sizes
        self.img_embedding_size = img_embedding_size
        self.encoder_num_convs = encoder_num_convs
        self.decoder_num_convs = decoder_num_convs
        self.encoder_conv_hidden_sizes = encoder_conv_hidden_sizes
        self.decoder_conv_hidden_sizes = decoder_conv_hidden_sizes
        self.ks = ks
        self.num_fcs = num_fcs
        self.num_encoder_fcs = num_encoder_fcs
        self.num_decoder_fcs = num_decoder_fcs
        self.fc_h_dim = fc_h_dim
        self.skip_layers = skip_layers
        self.lr = lr
        self.dataset_params = dataset_params
        self.activation = activation

        self.img_encoder = self._get_img_encoder()
        self.img_decoder = self._get_img_decoder()
        self.dyn_model = self._get_dyn_model()

        self.loss = None #TODO: Define the loss function
        self.mse_loss = nn.MSELoss()

        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'bubble_dynamics_residual_model'

    @property
    def name(self):
        return self.get_name()

    def _get_img_encoder(self):
        sizes = self._get_sizes()
        img_size = sizes['imprint']# (C_in, W_in, H_in)
        img_encoder = ImageEncoder(input_size=img_size,
                                   latent_size=self.img_embedding_size,
                                   num_convs=self.encoder_num_convs,
                                   conv_h_sizes=self.encoder_conv_hidden_sizes,
                                   ks=self.ks,
                                   num_fcs=self.num_encoder_fcs,
                                   fc_hidden_size=self.fc_h_dim,
                                   activation=self.activation)
        return img_encoder

    def _get_img_decoder(self):
        # TODO: Debug
        sizes = self._get_sizes()
        img_size = sizes['imprint']  # (C_in, W_in, H_in)
        img_decoder = ImageDecoder(output_size=img_size,
                                   latent_size=self.img_embedding_size,
                                   num_convs=self.decoder_num_convs,
                                   conv_h_sizes=self.decoder_conv_hidden_sizes,
                                   ks=self.ks,
                                   num_fcs=self.num_decoder_fcs,
                                   fc_hidden_size=self.fc_h_dim,
                                   activation=self.activation)
        return img_decoder

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        wrench_size = sizes['wrench']
        pos_size = sizes['position']
        quat_size = sizes['orientation']
        pose_size = pos_size + quat_size
        action_size = sizes['action']
        dyn_input_size = self.img_embedding_size + wrench_size + pose_size + action_size
        dyn_output_size = self.img_embedding_size + wrench_size + pose_size
        dyn_model_sizes = [dyn_input_size] + [self.fc_h_dim]*self.num_fcs + [dyn_output_size]
        dyn_model = FCModule(sizes=dyn_model_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return dyn_model

    def forward(self, imprint, wrench, position, orientation, action):
        sizes = self._get_sizes()
        wrench_size = sizes['wrench']
        position_size = sizes['position']
        quat_size = sizes['orientation']
        imprint_input_emb = self.img_encoder(imprint)
        dyn_input = torch.cat([imprint_input_emb, wrench, position, orientation, action], dim=-1)
        dyn_output = self.dyn_model(dyn_input)
        imprint_output_emb, wrench_delta, pos_delta, quat_delta = torch.split(dyn_output, [self.img_embedding_size, wrench_size, position_size, quat_size], dim=-1)
        imprint_delta = self.img_decoder(imprint_output_emb)
        return imprint_delta, wrench_delta, pos_delta, quat_delta

    def _get_sizes(self):
        imprint_size = self.input_sizes['init_imprint']
        wrench_size = np.prod(self.input_sizes['init_wrench'])
        pose_size = np.prod(self.input_sizes['init_pos'])
        quat_size = + np.prod(self.input_sizes['init_quat'])
        action_size = np.prod(self.input_sizes['action'])
        sizes = {'imprint': imprint_size,
                 'wrench': wrench_size,
                 'position': pose_size,
                 'orientation': quat_size,
                 'action': action_size
                 }
        return sizes

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx, phase='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch, batch_idx, phase='val')
        return loss
    
    def _step(self, batch, batch_idx, phase='train'):
        imprint_t = batch['init_imprint']
        wrench_t = batch['init_wrench']
        pos_t = batch['init_pos']
        quat_t = batch['init_quat']
        action = batch['action']
        imprint_d_gth = batch['delta_imprint']
        wrench_d_gth = batch['delta_wrench']
        pos_d_gth = batch['delta_pos']
        quat_d_gth = batch['delta_quat']

        imprint_delta, wrench_delta, pos_delta, quat_delta = self.forward(imprint_t, wrench_t, pos_t, quat_t, action)

        loss = self._compute_loss(imprint_delta, wrench_delta, pos_delta, quat_delta, imprint_d_gth, wrench_d_gth,
                                  pos_d_gth, quat_d_gth)

        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)

        predicted_grid = self._get_image_grid(imprint_delta)
        gth_grid = self._get_image_grid(imprint_d_gth)
        self.logger.experiment.add_image('delta_imprint_predicted_{}'.format(phase), predicted_grid, self.global_step)
        self.logger.experiment.add_image('delta_imprint_gt_{}'.format(phase), gth_grid, self.global_step)

    def _get_image_grid(self, batched_img):
        # swap the axis so the grid is (batch_size, num_channels, h, w)
        desired_shape = [x for x in batched_img.shape]
        desired_shape[-1] *= batched_img.shape[1]
        desired_shape[1] = 1
        grid_img = torchvision.utils.make_grid(batched_img.view(desired_shape))
        return grid_img
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _compute_loss(self, imprint_delta, wrench_delta, pos_delta, quat_delta, imprint_d_gth, wrench_d_gth, pos_d_gth, quat_d_gth): # TODO: Add inputs
        
        imprint_reconstruction_loss = self.mse_loss(imprint_delta, imprint_d_gth)
        wrench_loss = self.mse_loss(wrench_delta, wrench_d_gth)
        pos_loss = self.mse_loss(pos_delta, pos_d_gth)
        quat_loss = self.mse_loss(quat_delta, quat_d_gth) # TODO: Improve

        loss = imprint_reconstruction_loss + wrench_loss + pos_loss + quat_loss
        
        return loss



