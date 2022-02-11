import pdb

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


class BubbleDynamicsPretrainedAEModel(pl.LightningModule):
    """
    Model designed to model the bubbles dynamics.
    Given s_t, and a_t, it produces ∆s, where s_{t+1} = s_t + ∆s
     * Here s_t is composed by:
        - Depth image from each of the bubbles
    * The depth images are embedded into a vector which is later concatenated with the wrench and pose information
    """
    def __init__(self, input_sizes, load_autoencoder_version=31, num_fcs=2, fc_h_dim=100, skip_layers=None, lr=1e-4, dataset_params=None, load_norm=False, activation='relu'):
        super().__init__()
        self.input_sizes = input_sizes
        self.num_fcs = num_fcs
        self.fc_h_dim = fc_h_dim
        self.skip_layers = skip_layers
        self.lr = lr
        self.dataset_params = dataset_params
        self.activation = activation
        self.load_norm = load_norm

        self.autoencoder = self._load_autoencoder(load_version=load_autoencoder_version, data_path=dataset_params['data_name'])
        self.autoencoder.freeze()
        self.img_embedding_size = self.autoencoder.img_embedding_size # load it from the autoencoder

        self.dyn_model = self._get_dyn_model()

        self.mse_loss = nn.MSELoss()

        self.save_hyperparameters() # Important! Every model extension must add this line!

    @classmethod
    def get_name(cls):
        return 'bubble_dynamics_pretrained_autoencoder_model'

    @property
    def name(self):
        return self.get_name()

    def _load_autoencoder(self, load_version, data_path, load_epoch=None, load_step=None):
        Model = BubbleAutoEncoderModel
        model_name = Model.get_name()
        if load_epoch is None or load_step is None:
            version_chkp_path = os.path.join(data_path, 'tb_logs', '{}'.format(model_name),
                                             'version_{}'.format(load_version), 'checkpoints')
            checkpoints_fs = [f for f in os.listdir(version_chkp_path) if
                              os.path.isfile(os.path.join(version_chkp_path, f))]
            checkpoint_path = os.path.join(version_chkp_path, checkpoints_fs[0])
        else:
            checkpoint_path = os.path.join(data_path, 'tb_logs', '{}'.format(model_name),
                                           'version_{}'.format(load_version), 'checkpoints',
                                           'epoch={}-step={}.ckpt'.format(load_epoch, load_step))

        model = Model.load_from_checkpoint(checkpoint_path)

        return model

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        action_size = sizes['action']
        dyn_input_size = self.img_embedding_size + action_size
        dyn_output_size = self.img_embedding_size
        dyn_model_sizes = [dyn_input_size] + [self.fc_h_dim]*self.num_fcs + [dyn_output_size]
        dyn_model = FCModule(sizes=dyn_model_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return dyn_model

    def query(self, imprint, action):
        sizes = self._get_sizes()
        if self.load_norm:
            imprint_input_emb = self.autoencoder.img_encoder(imprint)
            dyn_input = torch.cat([imprint_input_emb, action], dim=-1)
            dyn_output_delta = self.dyn_model(dyn_input)
            imprint_output_emb = imprint_input_emb + dyn_output_delta
            imprint_next = self.autoencoder.img_decoder(imprint_output_emb)
        else:
            imprint_input_emb = self.autoencoder.encode(imprint)
            dyn_input = torch.cat([imprint_input_emb, action], dim=-1)
            dyn_output_delta = self.dyn_model(dyn_input)
            imprint_output_emb = imprint_input_emb + dyn_output_delta
            imprint_next = self.autoencoder.decode(imprint_output_emb)
        return imprint_next

    def forward(self, imprint, action):
        if self.load_norm:
            # apply the normalization to the imprint input, and then to the output, because the model has been trained on the normalized space.
            imprint_t = self.autoencoder._norm_imprint(imprint)
            imprint_next = self.query(imprint_t, action)
            imprint_next = self.autoencoder._denorm_imprint(imprint_next)
        else:
            imprint_next = self.query(imprint, action)
        return imprint_next

    def _get_sizes(self):
        imprint_size = self.input_sizes['init_imprint']
        wrench_size = np.prod(self.input_sizes['init_wrench'])
        pose_size = np.prod(self.input_sizes['init_pos'])
        quat_size = np.prod(self.input_sizes['init_quat'])
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
        imprint_next = batch['final_imprint']
        if self.load_norm:
            # normalize the tensors
            imprint_t = self.autoencoder._norm_imprint(imprint_t)
            imprint_next = self.autoencoder._norm_imprint(imprint_next)
        action = batch['action']

        imprint_next_rec = self.query(imprint_t, action)

        loss = self._compute_loss(imprint_next_rec, imprint_next)
        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)

        predicted_grid = self._get_image_grid(imprint_next_rec*torch.max(imprint_next_rec)/torch.max(imprint_next)) # trasform so they are in the same range
        gth_grid = self._get_image_grid(imprint_next)
        if batch_idx == 0:
            if self.current_epoch == 0:
                self.logger.experiment.add_image('init_imprint_{}'.format(phase), self._get_image_grid(imprint_t), self.global_step)
                self.logger.experiment.add_image('next_imprint_gt_{}'.format(phase), gth_grid, self.global_step)
            self.logger.experiment.add_image('next_imprint_predicted_{}'.format(phase), predicted_grid, self.global_step)
        return loss

    def _get_image_grid(self, batched_img, cmap='jet'):
        # reshape the batched_img to have the same imprints one above the other
        batched_img = batched_img.detach().cpu()
        batched_img_r = batched_img.reshape(*batched_img.shape[:1], -1, *batched_img.shape[3:]) # (batch_size, 2*W, H)
        # Add padding
        padding_pixels = 5
        batched_img_padded = F.pad(input=batched_img_r,
                                   pad=(padding_pixels, padding_pixels, padding_pixels, padding_pixels),
                                   mode='constant',
                                   value=0)
        batched_img_cmap = self._cmap_tensor(batched_img_padded, cmap=cmap) # size (..., w,h, 3)
        num_dims = len(batched_img_cmap.shape)
        grid_input = batched_img_cmap.permute(*np.arange(num_dims-3), -1, -3, -2)
        grid_img = torchvision.utils.make_grid(grid_input)
        return grid_img

    def _cmap_tensor(self, img_tensor, cmap='jet'):
        cmap = cm.get_cmap(cmap)
        mapped_img_ar = cmap(img_tensor/torch.max(img_tensor)) # (..,w,h,4)
        mapped_img_ar = mapped_img_ar[..., :3] # (..,w,h,3) -- get rid of the alpha value
        mapped_img = torch.tensor(mapped_img_ar).to(self.device)
        return mapped_img

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _compute_loss(self, imprint_rec, imprint_gth):
        imprint_reconstruction_loss = self.mse_loss(imprint_rec, imprint_gth)
        loss = imprint_reconstruction_loss
        return loss


class BubbleFullDynamicsPretrainedAEModel(BubbleDynamicsPretrainedAEModel):

    @classmethod
    def get_name(cls):
        return 'bubble_full_dynamics_pretrained_autoencoder_model'

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        action_size = sizes['action']
        dyn_input_size = self.img_embedding_size + action_size + sizes['wrench'] + sizes['position'] + sizes['orientation']
        dyn_output_size = self.img_embedding_size + sizes['wrench'] + sizes['position'] + sizes['orientation']
        dyn_model_sizes = [dyn_input_size] + [self.fc_h_dim]*self.num_fcs + [dyn_output_size]
        dyn_model = FCModule(sizes=dyn_model_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return dyn_model

    def query(self, imprint, wrench, pos, ori, action):
        sizes = self._get_sizes()
        # Encode imprint
        if self.load_norm:
            imprint_input_emb = self.autoencoder.img_encoder(imprint)
        else:
            imprint_input_emb = self.autoencoder.encode(imprint)

        state_dyn_input = torch.cat([imprint_input_emb, wrench, pos, ori], dim=-1)
        dyn_input = torch.cat([state_dyn_input, action], dim=-1)
        state_dyn_delta = self.dyn_model(dyn_input)
        state_dyn_output = state_dyn_input + state_dyn_delta
        imprint_output_emb, wrench_next, pos_next, ori_next = torch.split(state_dyn_output, (self.img_embedding_size, sizes['wrench'], sizes['position'], sizes['orientation']), dim=-1)

        # Decode imprint
        if self.load_norm:
            imprint_next = self.autoencoder.img_decoder(imprint_output_emb)
        else:
            imprint_next = self.autoencoder.decode(imprint_output_emb)

        return imprint_next, wrench_next, pos_next, ori_next

    def forward(self, imprint, wrench, pos, ori, action):
        if self.load_norm:
            # apply the normalization to the imprint input, and then to the output, because the model has been trained on the normalized space.
            imprint_t = self.autoencoder._norm_imprint(imprint)
            imprint_next, wrench_next, pos_next, ori_next = self.query(imprint_t, wrench, pos, ori, action)
            imprint_next = self.autoencoder._denorm_imprint(imprint_next)
        else:
            imprint_next, wrench_next, pos_next, ori_next = self.query(imprint, action)
        return imprint_next, wrench_next, pos_next, ori_next

    def _step(self, batch, batch_idx, phase='train'):
        imprint_t = batch['init_imprint']
        wrench_t = batch['init_wrench']
        pos_t = batch['init_pos']
        ori_t = batch['init_quat']
        imprint_next = batch['final_imprint']
        wrench_next = batch['final_wrench']
        pos_next = batch['final_pos']
        ori_next = batch['final_quat']
        if self.load_norm:
            # normalize the tensors
            imprint_t = self.autoencoder._norm_imprint(imprint_t)
            imprint_next = self.autoencoder._norm_imprint(imprint_next)
        action = batch['action']

        imprint_next_rec, wrench_next_rec, pos_next_rec, ori_next_rec = self.query(imprint_t, wrench_t, pos_t, ori_t, action)

        loss = self._compute_loss(imprint_next_rec, wrench_next_rec, pos_next_rec, ori_next_rec, imprint_next, wrench_next, pos_next, ori_next)

        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)

        predicted_grid = self._get_image_grid(imprint_next_rec*torch.max(imprint_next_rec)/torch.max(imprint_next)) # trasform so they are in the same range
        gth_grid = self._get_image_grid(imprint_next)
        if batch_idx == 0:
            if self.current_epoch == 0:
                self.logger.experiment.add_image('init_imprint_{}'.format(phase), self._get_image_grid(imprint_t), self.global_step)
                self.logger.experiment.add_image('next_imprint_gt_{}'.format(phase), gth_grid, self.global_step)
            self.logger.experiment.add_image('next_imprint_predicted_{}'.format(phase), predicted_grid, self.global_step)
        return loss

    def _compute_loss(self, imprint_rec, wrench_rec, pos_rec, ori_rec, imprint_gth, wrench_gth, pos_gth, ori_gth):
        imprint_reconstruction_loss = self.mse_loss(imprint_rec, imprint_gth)
        wrench_reconstruction_loss = self.mse_loss(wrench_rec, wrench_gth)
        pos_reconstruction_loss = self.mse_loss(pos_rec, pos_gth)
        ori_reconstruction_loss = self.mse_loss(ori_rec, ori_gth)
        loss = imprint_reconstruction_loss + 0.01 * (wrench_reconstruction_loss + pos_reconstruction_loss + ori_reconstruction_loss)
        return loss

