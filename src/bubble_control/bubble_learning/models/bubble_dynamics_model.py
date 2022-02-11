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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'bubble_dynamics_model'

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        dyn_input_size = sizes['dyn_input_size']
        dyn_output_size = sizes['dyn_output_size']
        dyn_model_sizes = [dyn_input_size] + [self.fc_h_dim] * self.num_fcs + [dyn_output_size]
        dyn_model = FCModule(sizes=dyn_model_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return dyn_model

    def _get_sizes(self):
        sizes = super()._get_sizes()
        dyn_input_size = self.img_embedding_size + sizes['wrench'] + self.object_embedding_size + sizes['position'] + sizes['orientation'] + sizes['action']
        dyn_output_size = self.img_embedding_size + sizes['wrench']
        sizes['dyn_input_size'] = dyn_input_size
        sizes['dyn_output_size'] = dyn_output_size
        return sizes

    def forward(self, imprint, wrench, object_model, pos, ori, action):
        sizes = self._get_sizes()
        imprint_input_emb = self.autoencoder.encode(imprint) # (B, imprint_emb_size)
        obj_model_emb = self.object_embedding_module(object_model) # (B, imprint_emb_size)
        state_dyn_input = torch.cat([imprint_input_emb, wrench], dim=-1)
        dyn_input = torch.cat([state_dyn_input, obj_model_emb, pos, ori, action], dim=-1)
        state_dyn_output_delta = self.dyn_model(dyn_input)
        state_dyn_output = state_dyn_input + state_dyn_output_delta
        imprint_emb_next, wrench_next = torch.split(state_dyn_output, (self.img_embedding_size, sizes['wrench']), dim=-1)
        imprint_next = self.autoencoder.decode(imprint)
        return imprint_next, wrench_next

    def _step(self, batch, batch_idx, phase='train'):
        imprint_t = batch['init_imprint']
        wrench_t = batch['init_wrench']
        pos_t = batch['init_pos']
        ori_t = batch['init_quat']
        object_model = batch['object_model']
        imprint_next = batch['final_imprint']
        wrench_next = batch['final_wrench']
        pos_next = batch['final_pos']
        ori_next = batch['final_quat']
        action = batch['action']

        imprint_next_rec, wrench_next_rec = self.forward(imprint_t, wrench_t, pos_t, ori_t, object_model, action)

        loss = self._compute_loss(imprint_next_rec, wrench_next_rec, imprint_next, wrench_next)

        # Log the results: -------------------------
        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)
        # Log imprints
        # TODO: Improve this --
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





