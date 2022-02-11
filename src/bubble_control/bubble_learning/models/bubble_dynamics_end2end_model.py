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
        self.img_encoder = self._get_img_encoder()
        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'bubble_end2end_model'

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

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        dyn_input_size = sizes['dyn_input_size']
        dyn_output_size = sizes['dyn_output_size']
        dyn_model_sizes = [dyn_input_size] + [self.fc_h_dim]*self.num_fcs + [dyn_output_size]
        dyn_model = FCModule(sizes=dyn_model_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return dyn_model

    def _get_sizes(self):
        sizes = super()._get_sizes()
        dyn_input_size = self.img_embedding_size + sizes['wrench'] + self.object_embedding_size + sizes['position'] + sizes['orientation'] + sizes['action']
        dyn_output_size = sizes['position'] + sizes['orientation']
        obj_pos_size = np.prod(self.input_sizes['obj_pos'])
        obj_quat_size = np.prod(self.input_sizes['obj_quat'])
        sizes['dyn_input_size'] = dyn_input_size
        sizes['dyn_output_size'] = dyn_output_size
        sizes['object_position'] = obj_pos_size
        sizes['object_orientation'] = obj_quat_size
        return sizes

    def forward(self, imprint, wrench, object_model, pos, ori, action):
        sizes = self._get_sizes()
        obj_pos_size = sizes['object_position']
        obj_quat_size = sizes['object_orientation']
        imprint_input_emb = self.img_encoder(imprint)
        obj_model_emb = self.object_embedding_module(object_model)  # (B, imprint_emb_size)
        dyn_input = torch.cat([imprint_input_emb, wrench, obj_model_emb, pos, ori, action], dim=-1)
        dyn_output = self.dyn_model(dyn_input)
        obj_pos, obj_quat = torch.split(dyn_output, [obj_pos_size, obj_quat_size], dim=-1)
        return obj_pos, obj_quat

    def _step(self, batch, batch_idx, phase='train'):
        imprint_t = batch['init_imprint']
        wrench_t = batch['init_wrench']
        pos_t = batch['init_pos']
        ori_t = batch['init_quat']
        action = batch['action']
        obj_pos_gth = batch['obj_pos']
        obj_ori_gth = batch['obj_quat']
        object_model = batch['object_model']
        obj_pos, obj_ori = self.forward(imprint_t, wrench_t, pos_t, ori_t, object_model, action)

        loss = self._compute_loss(obj_pos, obj_ori, obj_pos_gth, obj_ori_gth, object_model)

        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)

        return loss

    def _compute_loss(self, obj_pos, obj_ori, obj_pos_gth, obj_ori_gth, obj_model):
        # TODO: Compute the loss based on the object model.
        # self.pose_loss.model = obj_model
        # R_pred =
        # R_gth =
        # pose_loss = self.pose_loss(R_pred, obj_pos, R_gth, obj_pos_gth)
        # MSE Loss on position and orientation (encoded as aixis-angle 3 values)
        pose_loss = self.mse_loss(obj_pos, obj_pos_gth) + self.mse_loss(obj_ori, obj_ori_gth)
        return pose_loss