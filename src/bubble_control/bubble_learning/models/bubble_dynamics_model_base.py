import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib import cm
import torchvision
import abc


from bubble_control.bubble_learning.models.aux.fc_module import FCModule
from bubble_control.bubble_learning.models.aux.img_encoder import ImageEncoder
from bubble_control.bubble_learning.models.aux.img_decoder import ImageDecoder
from bubble_control.bubble_learning.models.bubble_autoencoder import BubbleAutoEncoderModel
from bubble_control.bubble_learning.models.pointnet.pointnet_loading_utils import get_pretrained_pointnet2_object_embeding


class BubbleDynamicsModelBase(pl.LightningModule):
    def __init__(self, input_sizes, load_autoencoder_version=31, object_embedding_size=10, num_fcs=2, fc_h_dim=100, skip_layers=None, lr=1e-4, dataset_params=None, load_norm=False, activation='relu', freeze_object_module=True):
        super().__init__()
        self.input_sizes = input_sizes
        self.object_embedding_size = object_embedding_size
        self.num_fcs = num_fcs
        self.fc_h_dim = fc_h_dim
        self.skip_layers = skip_layers
        self.lr = lr
        self.dataset_params = dataset_params
        self.activation = activation
        self.load_norm = load_norm
        self.freeze_object_module = freeze_object_module

        self.object_embedding_module = self._load_object_embedding_module(object_embedding_size=self.object_embedding_size, freeze=self.freeze_pointnet)
        self.autoencoder = self._load_autoencoder(load_version=load_autoencoder_version, data_path=dataset_params['data_name'])
        self.autoencoder.freeze()
        self.img_embedding_size = self.autoencoder.img_embedding_size # load it from the autoencoder

        self.dyn_model = self._get_dyn_model()

        self.mse_loss = nn.MSELoss()

        self.save_hyperparameters() # Important! Every model extension must add this line!

    @classmethod
    def get_name(cls):
        return 'bubble_dynamics_model_base'

    @property
    def name(self):
        return self.get_name()

    @abc.abstractmethod
    def forward(self, imprint, wrench, object_model, pos, ori, action):
        pass

    @abc.abstractmethod
    def _get_dyn_model(self):
        pass

    @abc.abstractmethod
    def _step(self, batch, batch_idx, phase='train'):
        pass

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx, phase='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch, batch_idx, phase='val')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

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
                 'action': action_size,
                 }
        return sizes

    def get_model_input(self, sample):
        input_key = self.get_input_keys()
        model_input = [sample[key] for key in input_key]
        model_input = tuple(model_input)
        return model_input

    def get_model_output(self, sample):
        output_keys = self.get_model_output_keys()
        next_state_map = self.get_next_state_map()
        model_output = [sample[next_state_map[key]] for key in output_keys]
        model_output = tuple(model_output)
        return model_output

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        dyn_input_size = sizes['dyn_input_size']
        dyn_output_size = sizes['dyn_output_size']
        dyn_model_sizes = [dyn_input_size] + [self.fc_h_dim]*self.num_fcs + [dyn_output_size]
        dyn_model = FCModule(sizes=dyn_model_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return dyn_model
    
    
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
    
    # Loading Functionalities: -----------------------------------------------------------------------------------------

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

    def _load_object_embedding_module(self, object_embedding_size, freeze=True):
        pointnet_model = get_pretrained_pointnet2_object_embeding(obj_embedding_size=object_embedding_size, freeze=freeze)
        # Expected input shape (BatchSize, NumPoints, NumChannels), where NumChannels=3 (xyz)
        return pointnet_model


    # AUX Functions: ---------------------------------------------------------------------------------------------------

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