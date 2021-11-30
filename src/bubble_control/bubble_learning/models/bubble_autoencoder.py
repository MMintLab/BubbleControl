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

from bubble_control.bubble_learning.models.bubble_dynamics_residual_model import BubbleDynamicsResidualModel


class BubbleAutoEncoderModel(BubbleDynamicsResidualModel):

    def __init__(self, *args, reconstruct_key='delta_imprint',**kwargs):
        self.reconstruct_key = reconstruct_key
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls):
        return 'bubble_autoencoder_model'

    def _get_dyn_model(self):
        return None # We do not have a dyn model in this case

    def forward(self, imprint):
        imprint_emb = self.img_encoder(imprint)
        imprint_reconstructed = self.img_decoder(imprint_emb)
        return imprint_reconstructed

    def _get_sizes(self):
        imprint_size = self.input_sizes[self.reconstruct_key]
        sizes = {'imprint': imprint_size}
        return sizes

    def _step(self, batch, batch_idx, phase='train'):
        imprint_t = batch[self.reconstruct_key]
        imprint_rec = self.forward(imprint_t)
        loss = self._compute_loss(imprint_t, imprint_rec)
        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)
        # add image:
        if batch_idx == 0:
            reconstructed_grid = self._get_image_grid(imprint_rec)
            gth_grid = self._get_image_grid(imprint_t)
            self.logger.experiment.add_image('{}_reconstructed_{}'.format(self.reconstruct_key, phase), reconstructed_grid, self.global_step)
            self.logger.experiment.add_image('{}_gth_{}'.format(self.reconstruct_key, phase), gth_grid, self.global_step)
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx, phase='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch, batch_idx, phase='val')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _compute_loss(self, imprint_rec, imprint_d_gth):
        imprint_reconstruction_loss = self.mse_loss(imprint_rec, imprint_d_gth)
        loss = imprint_reconstruction_loss

        return loss
