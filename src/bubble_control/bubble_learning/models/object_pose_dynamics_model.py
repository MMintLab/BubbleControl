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
import cv2
import pytorch3d.transforms as batched_trs

from bubble_control.bubble_learning.models.aux.fc_module import FCModule
from bubble_control.bubble_learning.models.aux.img_encoder import ImageEncoder
from bubble_control.bubble_learning.models.aux.img_decoder import ImageDecoder
from bubble_control.bubble_learning.models.bubble_autoencoder import BubbleAutoEncoderModel
from bubble_control.bubble_learning.models.dynamics_model_base import DynamicsModelBase
from bubble_control.bubble_learning.aux.orientation_trs import QuaternionToAxis
from bubble_control.bubble_learning.aux.pose_loss import ModelPoseLoss


class ObjectPoseDynamicsModel(DynamicsModelBase):

    def __init__(self, *args, num_to_log=40, input_batch_norm=True, **kwargs):
        self.num_to_log = num_to_log
        self.input_batch_norm = input_batch_norm
        super().__init__(*args, **kwargs)
        self.dyn_model = self._get_dyn_model()
        self.pose_loss = ModelPoseLoss()
        self.plane_normal = nn.Parameter(torch.tensor([1, 0, 0], dtype=torch.float), requires_grad=False)
        sizes = self._get_sizes()
        self.dyn_input_batch_norm = nn.BatchNorm1d(
            num_features=sizes['dyn_input_size'])  # call eval() to freeze the mean and std estimation
        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'object_pose_dynamics_model'

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

    def _get_dyn_input_size(self, sizes):
        dyn_input_size = sizes['init_object_pose'] + sizes['init_pos'] + sizes['init_quat'] + self.object_embedding_size + sizes['action']
        return dyn_input_size

    def _get_dyn_output_size(self, sizes):
        dyn_output_size = sizes['init_object_pose']
        return dyn_output_size

    def forward(self, obj_pose, pos, ori, object_model, action):
        # sizes = self._get_sizes()
        # obj_pos_size = sizes['object_position']
        # obj_quat_size = sizes['object_orientation']
        # obj_pose_size = obj_pos_size + obj_quat_size
        obj_model_emb = self.object_embedding_module(object_model)  # (B, imprint_emb_size)
        dyn_input = torch.cat([obj_pose, pos, ori, obj_model_emb, action], dim=-1)
        if self.input_batch_norm:
            dyn_input = self.dyn_input_batch_norm(dyn_input)
        dyn_output = self.dyn_model(dyn_input)
        obj_pose_next = dyn_output # we only predict object_pose
        return (obj_pose_next,)

    def get_state_keys(self):
        state_keys = ['init_object_pose', 'init_pos', 'init_quat', 'object_model']
        return state_keys
    
    def get_input_keys(self):
        input_keys = ['init_object_pose', 'init_pos', 'init_quat', 'object_model']
        return input_keys

    def get_model_output_keys(self):
        output_keys = ['init_object_pose']
        return output_keys

    def get_next_state_map(self):
        next_state_map = {
            'init_object_pose': 'final_object_pose'
        }
        return next_state_map

    def _compute_loss(self, obj_pose_pred, obj_pose_gth, object_model):
        # MSE Loss on position and orientation (encoded as aixis-angle 3 values)
        axis_angle_pred = obj_pose_pred[..., 3:]
        R_pred = batched_trs.axis_angle_to_matrix(axis_angle_pred)
        t_pred = obj_pose_pred[..., :3]
        axis_angle_gth = obj_pose_gth[..., 3:]
        R_gth = batched_trs.axis_angle_to_matrix(axis_angle_gth)
        t_gth = obj_pose_gth[..., :3]
        pose_loss = self.pose_loss(R_1=R_pred, t_1=t_pred, R_2=R_gth, t_2=t_gth, model_points=object_model)
        # pose_loss = self.mse_loss(obj_pose_pred, obj_pose_gth)
        loss = pose_loss
        return loss

    def _step(self, batch, batch_idx, phase='train'):
        action = batch['action']
        object_model = batch['object_model']

        model_input = self.get_model_input(batch)
        ground_truth = self.get_model_output(batch)
        model_output = self.forward(*model_input, action)
        loss = self._compute_loss(*model_output, *ground_truth, object_model)

        # Log the results: -------------------------
        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)
        # Log the images: -------------------------
        self._log_object_pose_images(obj_pose_pred=model_output[0][:self.num_to_log], obj_pose_gth=ground_truth[0][:self.num_to_log], phase=phase)
        return loss

    def _log_object_pose_images(self, obj_pose_pred, obj_pose_gth, phase):
        obj_trans_pred = obj_pose_pred[..., :3]
        obj_rot_pred = obj_pose_pred[..., 3:]
        obj_rot_angle_pred = self._get_angle_from_axis_angle(obj_rot_pred, self.plane_normal)
        obj_trans_gth = obj_pose_gth[..., :3]
        obj_rot_gth = obj_pose_gth[..., 3:]
        obj_rot_angle_gth = self._get_angle_from_axis_angle(obj_rot_gth, self.plane_normal)
        images = self._get_pose_images(obj_trans_pred, obj_rot_angle_pred, obj_trans_gth, obj_rot_angle_gth)
        grid = torchvision.utils.make_grid(images)
        self.logger.experiment.add_image('pose_estimation_{}'.format(phase), grid, self.global_step)

    def _get_angle_from_axis_angle(self, orientation, plane_normal):
        if orientation.shape[-1] == 4:
            q_to_ax = QuaternionToAxis()
            axis_angle = torch.from_numpy(q_to_ax._tr(orientation.detach().numpy()))
        else:
            axis_angle = orientation
        normal_axis_angle = torch.einsum('bi,i->b', axis_angle, plane_normal).unsqueeze(-1) * plane_normal.unsqueeze(0)
        angle = torch.norm(normal_axis_angle, dim=-1)
        return angle

    def _get_pose_images(self, trans_pred, rot_angle_pred, trans_gth, rot_angle_gth):
        images = []
        for i in range(len(trans_pred)):
            img = np.zeros([100, 100, 3], dtype=np.uint8)
            img.fill(100)
            pred_param = self._find_rect_param(trans_pred[i], rot_angle_pred[i], img)
            color_p = (255, 0, 0)
            self._draw_angled_rec(*pred_param, color_p, img)
            gth_param = self._find_rect_param(trans_gth[i], rot_angle_gth[i], img)
            color_gth = (0, 0, 255)
            self._draw_angled_rec(*gth_param, color_gth, img)
            img = torch.tensor(img)
            img = img.permute(2, 0, 1)
            images.append(img)
        return images

    def _find_rect_param(self, trans, rot, img):
        height = 0.06 * 100 / 0.15
        width = 0.015 * 100 / 0.15
        center_x = img.shape[0] / 2 + trans[0] * 10 / 0.15
        center_y = img.shape[1] / 2 + trans[1] * 10 / 0.15
        return center_x, center_y, width, height, rot.item()

    def _draw_angled_rec(self, x0, y0, width, height, angle, color, img):
        b = np.cos(angle) * 0.5
        a = np.sin(angle) * 0.5
        pt0 = (int(x0 - a * height - b * width),
            int(y0 + b * height - a * width))
        pt1 = (int(x0 + a * height - b * width),
            int(y0 - b * height - a * width))
        pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
        pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

        cv2.line(img, pt0, pt1, color, 3)
        cv2.line(img, pt1, pt2, color, 3)
        cv2.line(img, pt2, pt3, color, 3)
        cv2.line(img, pt3, pt0, color, 3)