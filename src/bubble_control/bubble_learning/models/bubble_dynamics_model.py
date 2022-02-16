import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib import cm
import torchvision
import numpy as np
import os
import sys

from bubble_control.bubble_learning.models.aux.fc_module import FCModule
from bubble_control.bubble_learning.models.aux.img_encoder import ImageEncoder
from bubble_control.bubble_learning.models.aux.img_decoder import ImageDecoder
from bubble_control.bubble_learning.models.bubble_autoencoder import BubbleAutoEncoderModel
from bubble_control.bubble_learning.models.bubble_dynamics_model_base import BubbleDynamicsModelBase


class BubbleDynamicsModel(BubbleDynamicsModelBase):

    def __init__(self, *args, input_batch_norm=True, **kwargs):
        self.input_batch_norm = input_batch_norm
        super().__init__(*args, **kwargs)
        sizes = self._get_sizes()
        self.dyn_input_batch_norm = nn.BatchNorm1d(num_features=sizes['dyn_input_size']) # call eval() to freeze the mean and std estimation
        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'bubble_dynamics_model'

    def _get_sizes(self):
        sizes = super()._get_sizes()
        dyn_input_size = self.img_embedding_size + sizes['wrench'] + self.object_embedding_size + sizes['position'] + sizes['orientation'] + sizes['action']
        dyn_output_size = self.img_embedding_size + sizes['wrench']
        sizes['dyn_input_size'] = dyn_input_size
        sizes['dyn_output_size'] = dyn_output_size
        return sizes

    def forward(self, imprint, wrench, pos, ori, object_model, action):
        sizes = self._get_sizes()
        imprint_input_emb = self.autoencoder.encode(imprint) # (B, imprint_emb_size)
        obj_model_emb = self.object_embedding_module(object_model) # (B, imprint_emb_size)
        state_dyn_input = torch.cat([imprint_input_emb, wrench], dim=-1)
        dyn_input = torch.cat([state_dyn_input, pos, ori, obj_model_emb, action], dim=-1)
        if self.input_batch_norm:
            dyn_input = self.dyn_input_batch_norm(dyn_input)
        state_dyn_output_delta = self.dyn_model(dyn_input)
        state_dyn_output = state_dyn_input + state_dyn_output_delta
        imprint_emb_next, wrench_next = torch.split(state_dyn_output, (self.img_embedding_size, sizes['wrench']), dim=-1)
        imprint_next = self.autoencoder.decode(imprint_emb_next)
        return imprint_next, wrench_next

    def get_state_keys(self):
        state_keys = ['init_imprint', 'init_wrench', 'init_pos', 'init_quat',
                      'object_model']
        return state_keys
    
    def get_input_keys(self):
        input_keys = ['init_imprint', 'init_wrench', 'init_pos', 'init_quat',
                      'object_model']
        return input_keys

    def get_model_output_keys(self):
        output_keys = ['init_imprint', 'init_wrench']
        return output_keys

    def get_next_state_map(self):
        next_state_map = {
            'init_imprint': 'final_imprint',
            'init_wrench': 'final_wrench',

        }
        return next_state_map

    def _step(self, batch, batch_idx, phase='train'):
        imprint_t = batch['init_imprint']
        imprint_next = batch['final_imprint']
        action = batch['action']

        model_input = self.get_model_input(batch)
        ground_truth = self.get_model_output(batch)

        model_output = self.forward(*model_input, action)

        loss = self._compute_loss(*model_output, *ground_truth)

        # Log the results: -------------------------
        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)
        # Log imprints
        # TODO: Improve this --
        imprint_indx = self.get_model_output_keys().index('init_imprint')
        imprint_next_rec = model_output[imprint_indx]
        predicted_grid = self._get_image_grid(imprint_next_rec * torch.max(imprint_next_rec) / torch.max(
            imprint_next))  # trasform so they are in the same range
        gth_grid = self._get_image_grid(imprint_next)
        if batch_idx == 0:
            if self.current_epoch == 0:
                self.logger.experiment.add_image('init_imprint_{}'.format(phase), self._get_image_grid(imprint_t),
                                                 self.global_step)
                self.logger.experiment.add_image('next_imprint_gt_{}'.format(phase), gth_grid, self.global_step)
            self.logger.experiment.add_image('next_imprint_predicted_{}'.format(phase), predicted_grid,
                                             self.global_step)
        return loss

    def _compute_loss(self, imprint_rec, wrench_rec, imprint_gth, wrench_gth):
        imprint_reconstruction_loss = self.mse_loss(imprint_rec, imprint_gth)
        wrench_reconstruction_loss = self.mse_loss(wrench_rec, wrench_gth)
        loss = imprint_reconstruction_loss + 0.01 * wrench_reconstruction_loss
        return loss





