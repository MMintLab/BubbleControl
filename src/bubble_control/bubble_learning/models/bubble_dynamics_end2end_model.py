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


class BubbleEnd2EndDynamicsModel(BubbleDynamicsModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_encoder = self._get_img_encoder()
        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'bubble_end2end_dynamics_model'

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

    def _get_sizes(self):
        sizes = super()._get_sizes()
        obj_pos_size = np.prod(self.input_sizes['obj_pos'])
        obj_quat_size = np.prod(self.input_sizes['obj_quat'])
        sizes['object_position'] = obj_pos_size
        sizes['object_orientation'] = obj_quat_size
        dyn_input_size = self.img_embedding_size + sizes['wrench'] + sizes['position'] + sizes['orientation'] + self.object_embedding_size + sizes['action']
        dyn_output_size = self.img_embedding_size + sizes['wrench'] + sizes['object_position'] + sizes['object_orientation']
        sizes['dyn_input_size'] = dyn_input_size
        sizes['dyn_output_size'] = dyn_output_size
        return sizes

    def forward(self, imprint, wrench, pos, ori, object_model, action):
        sizes = self._get_sizes()
        obj_pos_size = sizes['object_position']
        obj_quat_size = sizes['object_orientation']
        obj_pose_size = obj_pos_size + obj_quat_size
        wrench_size = sizes['wrench']
        imprint_input_emb = self.img_encoder(imprint)
        obj_model_emb = self.object_embedding_module(object_model)  # (B, imprint_emb_size)
        dyn_input = torch.cat([imprint_input_emb, wrench, pos, ori, obj_model_emb, action], dim=-1)
        dyn_output = self.dyn_model(dyn_input)
        imprint_emb_next, wrench_next, obj_pose_next  = torch.split(dyn_output, [self.img_embedding_size, wrench_size, obj_pose_size], dim=-1)
        imprint_next = self.autoencoder.decode(imprint_emb_next)        
        return imprint_next, wrench_next, obj_pose_next

    def get_state_keys(self):
        state_keys = ['init_imprint', 'init_wrench', 'init_pos', 'init_quat',
                      'object_model', 'init_object_pose']
        return state_keys
    
    def get_input_keys(self):
        input_keys = ['init_imprint', 'init_wrench', 'init_pos', 'init_quat',
                      'object_model']
        return input_keys

    def get_model_output_keys(self):
        output_keys = ['init_imprint', 'init_wrench', 'init_object_pose']
        return output_keys

    def get_next_state_map(self):
        next_state_map = {
            'init_imprint': 'final_imprint',
            'init_wrench': 'final_wrench',
            'init_object_pose': 'final_object_pose'
        }
        return next_state_map

    def _compute_loss(self, imprint_pred, wrench_pred, obj_pose_pred, imprint_gth, wrench_gth, obj_pose_gth):
        # MSE Loss on position and orientation (encoded as aixis-angle 3 values)
        pose_loss = self.mse_loss(obj_pose_pred, obj_pose_gth)
        imprint_loss = self.mse_loss(imprint_pred, imprint_gth)
        wrench_loss = self.mse_loss(wrench_pred, wrench_gth)
        loss = imprint_loss + wrench_loss + pose_loss # TODO: Consider adding different weights
        return loss