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


class BubbleEnd2EndModel(pl.LightningModule):

    def __init__(self, input_sizes, img_embedding_size=10, encoder_num_convs=3, encoder_conv_hidden_sizes=None, decoder_conv_hidden_sizes=None, ks=3, num_fcs=3, num_encoder_fcs=1, fc_h_dim=100, skip_layers=None, lr=1e-4, dataset_params=None, activation='relu'):
        super().__init__()
        self.input_sizes = input_sizes
        self.img_embedding_size = img_embedding_size
        self.encoder_num_convs = encoder_num_convs
        self.encoder_conv_hidden_sizes = encoder_conv_hidden_sizes
        self.decoder_conv_hidden_sizes = decoder_conv_hidden_sizes
        self.ks = ks
        self.num_fcs = num_fcs
        self.num_encoder_fcs = num_encoder_fcs
        self.fc_h_dim = fc_h_dim
        self.skip_layers = skip_layers
        self.lr = lr
        self.dataset_params = dataset_params
        self.activation = activation

        self.img_encoder = self._get_img_encoder()
        self.dyn_model = self._get_dyn_model()
        self.loss = None  # TODO: Define the loss function
        self.mse_loss = nn.MSELoss()

        self.save_hyperparameters()  # Important! Every model extension must add this line!

    @classmethod
    def get_name(cls):
        return 'bubble_end2end_model'

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

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        wrench_size = sizes['wrench']
        pos_size = sizes['position']
        quat_size = sizes['orientation']
        obj_pos_size = sizes['object_position']
        obj_quat_size = sizes['object_orientation']
        pose_size = pos_size + quat_size
        action_size = sizes['action']
        dyn_input_size = self.img_embedding_size + wrench_size + pose_size + action_size
        dyn_output_size = obj_pos_size + obj_quat_size
        dyn_model_sizes = [dyn_input_size] + [self.fc_h_dim]*self.num_fcs + [dyn_output_size]
        dyn_model = FCModule(sizes=dyn_model_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return dyn_model

    def _get_sizes(self):
        imprint_size = self.input_sizes['init_imprint']
        wrench_size = np.prod(self.input_sizes['init_wrench'])
        pose_size = np.prod(self.input_sizes['init_pos'])
        quat_size = np.prod(self.input_sizes['init_quat'])
        action_size = np.prod(self.input_sizes['action'])
        obj_pos_size = np.prod(self.input_sizes['obj_pos'])
        obj_quat_size = np.prod(self.input_sizes['obj_quat'])
        sizes = {'imprint': imprint_size,
                 'wrench': wrench_size,
                 'position': pose_size,
                 'orientation': quat_size,
                 'action': action_size,
                 'object_position': obj_pos_size,
                 'object_orientation': obj_quat_size,
                 }
        return sizes

    def forward(self, imprint, wrench, position, orientation, action):
        sizes = self._get_sizes()
        obj_pos_size = sizes['object_position']
        obj_quat_size = sizes['object_orientation']
        # TODO: Add the object model.
        imprint_input_emb = self.img_encoder(imprint)
        dyn_input = torch.cat([imprint_input_emb, wrench, position, orientation, action], dim=-1)
        dyn_output = self.dyn_model(dyn_input)
        obj_pos, obj_quat = torch.split(dyn_output, [obj_pos_size, obj_quat_size], dim=-1)
        return obj_pos, obj_quat

    def _step(self, batch, batch_idx, phase='train'):
        imprint_t = batch['init_imprint']
        wrench_t = batch['init_wrench']
        pos_t = batch['init_pos']
        quat_t = batch['init_quat']
        action = batch['action']
        obj_pos_gth = batch['obj_pos']
        obj_ori_gth = batch['obj_quat']
        obj_pos, obj_ori = self.forward(imprint_t, wrench_t, pos_t, quat_t, action)

        loss = self._compute_loss(obj_pos, obj_ori, obj_pos_gth, obj_ori_gth)

        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)

        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx, phase='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch, batch_idx, phase='val')
        return loss

    def _compute_loss(self, obj_pose, obj_ori, obj_pose_gth, obj_ori_gth):
        # TODO: Compute the loss based on the object model.
        pos_loss = self.mse_loss(obj_pose, obj_pose_gth)
        return pos_loss