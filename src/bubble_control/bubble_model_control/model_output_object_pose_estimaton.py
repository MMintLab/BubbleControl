import numpy as np
import os
import sys
import torch
import rospy
import tf2_ros as tf
import tf.transformations as tr
from geometry_msgs.msg import TransformStamped

from bubble_control.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from bubble_control.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from bubble_control.bubble_learning.aux.img_trs.block_upsampling_tr import BlockUpSamplingTr
from bubble_control.bubble_learning.models.bubble_dynamics_pretrained_ae_model import BubbleDynamicsPretrainedAEModel
from bubble_control.aux.load_confs import load_bubble_reconstruction_params
from bubble_control.bubble_pose_estimation.bubble_pc_reconstruction import BubblePCReconstructorOfflineDepth
from bubble_utils.bubble_tools.bubble_img_tools import unprocess_bubble_img


class ModelOutputObjectPoseEstimation(object):

    def __init__(self, object_name='marker'):
        self.object_name = object_name
        self.reconstruction_params = load_bubble_reconstruction_params()
        self.object_params = self.reconstruction_params[self.object_name]
        self.imprint_threshold = self.object_params['imprint_th']['depth']
        self.icp_threshold = self.object_params['icp_th']

        self.reconstructor = self._get_reconstructor()

    def _get_reconstructor(self):
        reconstructor = BubblePCReconstructorOfflineDepth(threshold=self.imprint_threshold,
                                                          object_name=self.object_name,
                                                          estimation_type='icp2d')
        return reconstructor

    def estimate_pose(self, sample):
        camera_info_r = sample['camera_info_r']
        camera_info_l = sample['camera_info_l']
        all_tfs = sample['all_tfs']
        # obtain reference (undeformed) depths
        ref_depth_img_r = sample['undef_depth_l'].squeeze()
        ref_depth_img_l = sample['undef_depth_l'].squeeze()

        predicted_imprint = sample['next_imprint']
        imprint_pred_r, imprint_pred_l = predicted_imprint

        # unprocess the imprints (add padding to move them back to the original shape)
        imprint_pred_r = unprocess_bubble_img(imprint_pred_r)
        imprint_pred_l = unprocess_bubble_img(imprint_pred_l)

        deformed_depth_r = ref_depth_img_r - imprint_pred_r  # CAREFUL: Imprint is defined as undef_depth_img - def_depth_img
        deformed_depth_l = ref_depth_img_l - imprint_pred_l

        # THIS hacks the ways to obtain data for the reconstructor
        self.reconstructor.references['left'] = ref_depth_img_l
        self.reconstructor.references['right'] = ref_depth_img_r
        self.reconstructor.depth_r = {'img': deformed_depth_r, 'frame': 'pico_flexx_right_optical_frame'}
        self.reconstructor.depth_l = {'img': deformed_depth_l, 'frame': 'pico_flexx_left_optical_frame'}
        self.reconstructor.camera_info['right'] = camera_info_r
        self.reconstructor.camera_info['left'] = camera_info_l
        # compute transformations from camera frames to grasp frame and transform the
        self.reconstructor.add_tfs(all_tfs)
        # estimate pose
        estimated_pose = self.reconstructor.estimate_pose(self.icp_threshold)
        # transform it to pos, quat instead of matrix
        estimated_pos = estimated_pose[:3,3]
        estimated_quat = tr.quaternion_from_matrix(estimated_pose)
        estimated_pose = np.concatenate([estimated_pos, estimated_quat])
        return estimated_pose


class ModelDownsampledOutputObjectPoseEstimation(ModelOutputObjectPoseEstimation):
    """
    Add the upsamplnig of the imprint to the esitimate pose query
    """

    def __init__(self, *args, factor_x=7, factor_y=7, method='bilinear', **kwargs):
        self.block_upsample_tr = BlockUpSamplingTr(factor_x=factor_x, factor_y=factor_y, method=method, keys_to_tr=['next_imprint'])
        super().__init__(*args, **kwargs)

    def estimate_pose(self, sample):
        # upsample the imprints
        sample_up = self._upsample_sample(sample)
        return super().estimate_pose(sample_up)

    def _upsample_sample(self, sample):
        # Upsample ouptut
        sample_up = self.block_upsample_tr(sample)
        return sample_up
