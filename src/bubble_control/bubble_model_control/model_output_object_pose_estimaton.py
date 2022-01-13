import numpy as np
import os
import sys
import torch
import pytorch3d.transforms as batched_trs
import rospy
import tf2_ros as tf
from abc import abstractmethod
import tf.transformations as tr
from geometry_msgs.msg import TransformStamped

from bubble_control.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from bubble_control.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from bubble_control.bubble_learning.aux.img_trs.block_upsampling_tr import BlockUpSamplingTr
from bubble_control.bubble_learning.models.bubble_dynamics_pretrained_ae_model import BubbleDynamicsPretrainedAEModel
from bubble_control.aux.load_confs import load_bubble_reconstruction_params, load_object_models
from bubble_control.bubble_pose_estimation.bubble_pc_reconstruction import BubblePCReconstructorOfflineDepth
from bubble_utils.bubble_tools.bubble_img_tools import unprocess_bubble_img

from bubble_control.bubble_pose_estimation.batched_pytorch_icp import icp_2d_masked, pc_batched_tr
from mmint_camera_utils.camera_utils import project_depth_image
from mmint_camera_utils.point_cloud_utils import project_pc, get_projection_tr
from bubble_utils.bubble_tools.bubble_pc_tools import get_imprint_mask


class ModelOutputObjectPoseEstimationBase(object):
    def __init__(self, object_name='marker', factor_x=1, factor_y=1, method='bilinear'):
        self.object_name = object_name
        self.reconstruction_params = load_bubble_reconstruction_params()
        self.object_params = self.reconstruction_params[self.object_name]
        self.imprint_threshold = self.object_params['imprint_th']['depth']
        self.icp_threshold = self.object_params['icp_th']
        self.block_upsample_tr = BlockUpSamplingTr(factor_x=factor_x, factor_y=factor_y, method=method,
                                                   keys_to_tr=['next_imprint'])

    def estimate_pose(self, sample):
        # upsample the imprints
        sample_up = self._upsample_sample(sample)
        return self._estimate_pose(sample_up)

    @abstractmethod
    def _estimate_pose(self, sample):
        # Return estimated object pose [x, y, z, qx, qy, qz, qw]
        pass

    def _upsample_sample(self, sample):
        # Upsample output
        sample_up = self.block_upsample_tr(sample)
        return sample_up


class BatchedModelOutputObjectPoseEstimation(ModelOutputObjectPoseEstimationBase):
    """ Work with pytorch tensors"""
    def __init__(self, *args, device=None, **kwargs):
        if device is None:
            device = torch.device('cpu')
        self.device = device
        super().__init__(*args, **kwargs)
        self.model_pcs = load_object_models()

    def _estimate_pose(self, batched_sample):
        """
        Estimate the object pose from the imprints using icp 2D. We compute it in parallel on batched operations
        :param sample: The sample is expected to be batched, i.e. all values (batch_size, original_size_1, ..., original_size_n)
        :return:
        """
        all_tfs = batched_sample['all_tfs']

        # Get imprints from sample
        predicted_imprint = batched_sample['next_imprint']
        imprint_pred_r = predicted_imprint[:,0]
        imprint_pred_l = predicted_imprint[:,1]

        # unprocess the imprints (add padding to move them back to the original shape)
        imprint_pred_r = unprocess_bubble_img(imprint_pred_r.unsqueeze(-1)).squeeze(-1) # ref frame:  -- (N, w, h)
        imprint_pred_l = unprocess_bubble_img(imprint_pred_l.unsqueeze(-1)).squeeze(-1) # ref frame:  -- (N, w, h)
        imprint_frame_r = 'pico_flexx_right_optical_frame'
        imprint_frame_l = 'pico_flexx_left_optical_frame'

        depth_ref_r = batched_sample['undef_depth_r'].squeeze() # (N, w, h)
        depth_ref_l = batched_sample['undef_depth_l'].squeeze() # (N, w, h)
        depth_def_r = depth_ref_r - imprint_pred_r  # CAREFUL: Imprint is defined as undef_depth_img - def_depth_img
        depth_def_l = depth_ref_l - imprint_pred_l  # CAREFUL: Imprint is defined as undef_depth_img - def_depth_img

        # Project imprints to get point coordinates
        Ks_r = batched_sample['camera_info_r']['K']
        Ks_l = batched_sample['camera_info_l']['K']
        pc_r = project_depth_image(depth_def_r, Ks_r)# (N, w, h, n_coords) -- n_coords=3
        pc_l = project_depth_image(depth_def_l, Ks_l)# (N, w, h, n_coords) -- n_coords=3

        # Compute mask -- filter out points
        imprint_threshold = self.imprint_threshold
        mask_r = get_imprint_mask(depth_ref_r, depth_def_r, imprint_threshold)
        mask_l = get_imprint_mask(depth_ref_l, depth_def_l, imprint_threshold)
        mask_r = torch.tensor(mask_r).to(predicted_imprint)
        mask_l = torch.tensor(mask_l).to(predicted_imprint)
        
        # Convert imprint point coordinates to grasp frame
        gf_X_ifr = self._get_transformation_matrix(all_tfs, 'grasp_frame', imprint_frame_r)
        gf_X_ifl = self._get_transformation_matrix(all_tfs, 'grasp_frame', imprint_frame_l)
        pc_shape = pc_r.shape
        pc_r_gf = pc_batched_tr(pc_r.view((pc_shape[0],-1,pc_shape[-1])), gf_X_ifr[..., :3, :3], gf_X_ifr[..., :3, 3]).view(pc_shape)
        pc_l_gf = pc_batched_tr(pc_l.view((pc_shape[0],-1,pc_shape[-1])), gf_X_ifl[..., :3, :3], gf_X_ifl[..., :3, 3]).view(pc_shape)
        pc_gf = torch.stack([pc_r_gf, pc_l_gf], dim=1) # (N, n_impr, w, h, n_coords)

        # Load object model model
        model_pc = np.asarray(self.model_pcs[self.object_name].points)
        model_pc = torch.tensor(model_pc).to(predicted_imprint.device)

        # Project points to 2d
        projection_axis = (1, 0, 0)
        projection_tr = torch.tensor(get_projection_tr(projection_axis)) # (4,4)
        pc_gf_projected = project_pc(pc_gf, projection_axis) # (N, n_impr, w, h, n_coords)
        pc_gf_2d = pc_gf_projected[..., :2] # only 2d coordinates
        pc_model_projected = project_pc(model_pc, projection_axis).unsqueeze(0).repeat_interleave(pc_gf.shape[0],dim=0)

        # Apply ICP 2d
        num_iterations = 20
        pc_scene = pc_gf_2d # pc_scene: (N, n_impr, w, h, n_coords)
        pc_scene_mask = torch.stack([mask_r, mask_l], dim=1) # (N, n_impr, w, h)
        pc_scene_mask = pc_scene_mask.unsqueeze(-1).repeat_interleave(2, dim=-1) # (N, n_impr, w, h, n_coords)
        pc_model_projected_2d = pc_model_projected[...,:2] # pc_model: (N, n_model_points, n_coords)
        
        # Apply ICP:
        device = self.device
        # TODO: Improve this filtering fuctions:
        pc_model_projected_2d = pc_model_projected_2d[:,:100,:] # TODO: Find a better way to downsample the model
        pc_scene = pc_scene[:, :, ::5, ::5, :]
        pc_scene_mask = pc_scene_mask[:, :, ::5, ::5, :]

        pc_model_projected_2d = pc_model_projected_2d.type(torch.float).to(device)
        pc_scene = pc_scene.type(torch.float).to(device)
        pc_scene_mask = pc_scene_mask.to(device)
        
        # TODO: Convert from float32 to float64
        Rs, ts = icp_2d_masked(pc_model_projected_2d, pc_scene, pc_scene_mask, num_iter=num_iterations)
        Rs = Rs.to(torch.device('cpu'))
        ts = ts.to(torch.device('cpu'))
        
        # Obtain object pose in grasp frame
        projected_ic_tr = torch.zeros(ts.shape[:-1]+(4, 4))
        projected_ic_tr[..., :2, :2] = Rs
        projected_ic_tr[..., :2,  3] = ts
        projection_tr = projection_tr.type(torch.float)
        unproject_tr = torch.linalg.inv(projection_tr)
        gf_X_objpose = torch.einsum('ji,kil->kjl', unproject_tr, torch.einsum('kij,jl->kil', projected_ic_tr, projection_tr))

        # Compute object pose in world frame
        wf_X_gf = self._get_transformation_matrix(all_tfs, 'med_base', 'grasp_frame').type(gf_X_objpose.dtype)
        wf_X_objpose = wf_X_gf @ gf_X_objpose

        # convert it to pose format [xs, ys, zs, qxs, qyx, qzs, qws]
        estimated_pos = wf_X_objpose[..., :3, 3]
        _estimated_quat = batched_trs.matrix_to_quaternion(wf_X_objpose[..., :3, :3]) # (qw,qx,qy,qz)
        estimated_quat = torch.index_select(_estimated_quat, dim=1, index=torch.LongTensor([1, 2, 3, 0]))# (qx,qy,qz,qw)
        estimated_poses = torch.cat([estimated_pos, estimated_quat], dim=-1)
        return estimated_poses

    def _get_transformation_matrix(self, all_tfs, source_frame, target_frame):
        w_X_sf = all_tfs[source_frame]
        w_X_tf = all_tfs[target_frame]
        sf_X_w = torch.linalg.inv(w_X_sf)
        sf_X_tf = sf_X_w @ w_X_tf
        return sf_X_tf


class ModelOutputObjectPoseEstimation(ModelOutputObjectPoseEstimationBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstructor = self._get_reconstructor()

    def _get_reconstructor(self):
        reconstructor = BubblePCReconstructorOfflineDepth(threshold=self.imprint_threshold,
                                                          object_name=self.object_name,
                                                          estimation_type='icp2d')
        return reconstructor

    def _estimate_pose(self, sample):
        camera_info_r = sample['camera_info_r']
        camera_info_l = sample['camera_info_l']
        all_tfs = sample['all_tfs']
        # obtain reference (undeformed) depths
        ref_depth_img_r = sample['undef_depth_r'].squeeze()
        ref_depth_img_l = sample['undef_depth_l'].squeeze()

        predicted_imprint = sample['next_imprint']
        imprint_pred_r, imprint_pred_l = predicted_imprint

        # unprocess the imprints (add padding to move them back to the original shape)
        imprint_pred_r = unprocess_bubble_img(np.expand_dims(imprint_pred_r,-1)).squeeze(-1)
        imprint_pred_l = unprocess_bubble_img(np.expand_dims(imprint_pred_l,-1)).squeeze(-1)

        deformed_depth_r = ref_depth_img_r - imprint_pred_r  # CAREFUL: Imprint is defined as undef_depth_img - def_depth_img
        deformed_depth_l = ref_depth_img_l - imprint_pred_l

        # THIS hacks the ways to obtain data for the reconstructor
        ref_depth_img_l = np.expand_dims(ref_depth_img_l, -1)
        ref_depth_img_r = np.expand_dims(ref_depth_img_r, -1)
        deformed_depth_r = np.expand_dims(deformed_depth_r, -1)
        deformed_depth_l = np.expand_dims(deformed_depth_l, -1)
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



