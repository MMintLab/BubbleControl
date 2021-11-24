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
from bubble_control.bubble_learning.models.bubble_dynamics_residual_model import BubbleDynamicsResidualModel
from bubble_control.bubble_learning.models.bubble_autoencoder import BubbleAutoEncoderModel


class BubblePCADynamicsResidualModel(BubbleDynamicsResidualModel):
    """
    Use as encoder and decoder pretrained bubble_autoencoder modules
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls):
        return 'bubble_pca_dynamics_residual_model'

    def _get_img_encoder(self):
        return None

    def _get_img_decoder(self):
        return None

    def forward(self, imprint, wrench, position, orientation, action):
        sizes = self._get_sizes()
        wrench_size = sizes['wrench']
        position_size = sizes['position']
        quat_size = sizes['orientation']
        # project the imprint using pca:
        imprint_reshaped = imprint.view(*imprint.shape[:1], -1, *imprint.shape[3:])
        U, S, V = torch.pca_lowrank(imprint_reshaped, q=self.img_embedding_size)
        dyn_input = torch.cat([S, wrench, position, orientation, action], dim=-1)
        dyn_output = self.dyn_model(dyn_input)
        imprint_output_emb, wrench_delta, pos_delta, quat_delta = torch.split(dyn_output,
                                                                              [self.img_embedding_size, wrench_size,
                                                                               position_size, quat_size], dim=-1)
        imprint_delta_reshaped = torch.bmm(U, torch.bmm(torch.diag_embed(imprint_output_emb), V.transpose(-1,-2)))
        imprint_delta = imprint_delta_reshaped.view(*imprint.shape)
        return imprint_delta, wrench_delta, pos_delta, quat_delta