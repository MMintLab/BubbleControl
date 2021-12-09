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
from bubble_control.bubble_pose_estimation.bubble_pc_reconstruction import BubblePCReconstructorBase
from bubble_utils.bubble_tools.bubble_img_tools import unprocess_bubble_img
from bubble_utils.bubble_tools.bubble_pc_tools import get_imprint_pc
from mmint_camera_utils.point_cloud_utils import tr_pointcloud


class BubblePCReconstructorOfflineDepth(BubblePCReconstructorBase):
    def __init__(self, *args, **kwargs):
        self.depth_r = {
            'img': None,
            'frame': None,
        }
        self.depth_l = {
            'img': None,
            'frame': None,
        }
        self.camera_info = {
            'left': None,
            'right': None,
        }
        self.buffer = tf.BufferCore()

        super().__init__(*args, **kwargs)

    def reference(self):
        pass

    def add_tfs(self, tfs_df):
        for indx, row in tfs_df.iterrows():
            # pack the tf into a TrasformStamped message
            q_i = [row['qx'], row['qy'], row['qz'], row['qw']]
            t_i = [row['x'], row['y'], row['z']]
            parent_frame_id = row['parent_frame']
            child_frame_id = row['child_frame']
            ts_msg_i = self._pack_transform_stamped_msg(q_i, t_i, parent_frame_id=parent_frame_id, child_frame_id=child_frame_id)
            self.buffer.set_transform(ts_msg_i, 'default_authority')

    def _tr_pc(self, pc, origin_frame, target_frame):
        ts_msg = self.buffer.lookup_transform_core(origin_frame, target_frame, rospy.Time(0))
        t, R = self._unpack_transform_stamped_msg(ts_msg)
        pc_tr = tr_pointcloud(pc, R, t)
        return pc_tr

    def _unpack_transform_stamped_msg(self, ts_msg):
        x = ts_msg.transform.translation.x
        y = ts_msg.transform.translation.y
        z = ts_msg.transform.translation.z
        qx = ts_msg.transform.rotation.x
        qy = ts_msg.transform.rotation.y
        qz = ts_msg.transform.rotation.z
        qw = ts_msg.transform.rotation.w
        q = np.array([qx, qy, qz, qw])
        t = np.array([x, y, z])
        R = tr.quaternion_matrix(q)[:3,:3]
        return t, R

    def _pack_transform_stamped_msg(self, q, t, parent_frame_id, child_frame_id):
        ts_msg = TransformStamped()
        ts_msg.header.stamp = rospy.Time(0)
        ts_msg.header.frame_id = parent_frame_id
        ts_msg.child_frame_id = child_frame_id
        ts_msg.transform.translation.x = t[0]
        ts_msg.transform.translation.y = t[1]
        ts_msg.transform.translation.z = t[2]
        ts_msg.transform.rotation.x = q[0]
        ts_msg.transform.rotation.y = q[1]
        ts_msg.transform.rotation.z = q[2]
        ts_msg.transform.rotation.w = q[3]
        return ts_msg

    def get_imprint(self, view=False):
        # return the contact imprint
        depth_r = self.depth_r['img']
        depth_l = self.depth_l['img']
        imprint_r = get_imprint_pc(self.references['right'], depth_r, threshold=self.threshold,
                                   K=self.camera_info['right']['K'])
        imprint_l = get_imprint_pc(self.references['left'], depth_l, threshold=self.threshold,
                                   K=self.camera_info['left']['K'])
        frame_r = self.depth_r['frame']
        frame_l = self.depth_l['frame']

        filtered_imprint_r = self.filter_pc(imprint_r)
        filtered_imprint_l = self.filter_pc(imprint_l)

        # trasform imprints
        imprint_r = self._tr_pc(filtered_imprint_r, frame_r, self.reconstruction_frame)
        imprint_l = self._tr_pc(filtered_imprint_l, frame_l, self.reconstruction_frame)

        imprint = np.concatenate([imprint_r, imprint_l], axis=0)
        return imprint


if __name__ == '__main__':

    data_name = '/home/mik/Desktop/drawing_data_cartesian'
    dataset = BubbleDrawingDataset(data_name=data_name, wrench_frame='med_base', tf_frame='grasp_frame')
    Model = BubbleDynamicsPretrainedAEModel

    load_version = 11

    block_downsample_tr = BlockDownSamplingTr(factor_x=7, factor_y=7, reduction='max', keys_to_tr=['init_imprint'])
    block_upsample_tr = BlockUpSamplingTr(factor_x=7, factor_y=7, method='bilinear')

    # load model:
    model_name = Model.get_name()
    version_chkp_path = os.path.join(data_name, 'tb_logs', '{}'.format(model_name),
                                     'version_{}'.format(load_version), 'checkpoints')
    checkpoints_fs = [f for f in os.listdir(version_chkp_path) if
                      os.path.isfile(os.path.join(version_chkp_path, f))]
    checkpoint_path = os.path.join(version_chkp_path, checkpoints_fs[0])

    model = Model.load_from_checkpoint(checkpoint_path)

    # load one sample:
    sample = dataset[0]

    # Downsample
    sample['init_imprint'] = sample['init_imprint']
    sample_down = block_downsample_tr(sample)

    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Query model   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    imprint_t = torch.tensor(sample_down['init_imprint']).unsqueeze(0).to(dtype=torch.float)
    action_t = torch.tensor(sample_down['action']).unsqueeze(0).to(dtype=torch.float)

    # predict next imprint
    next_imprint = model(imprint_t, action_t).squeeze()

    next_imprint = next_imprint.cpu().detach().numpy()
    
    # Upsample ouptut
    sample_out = {
        'next_imprint':next_imprint,
    }
    sample_up = block_upsample_tr(sample_out)

    predicted_imprint = sample_up['next_imprint']
    imprint_pred_r, imprint_pred_l = predicted_imprint

    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Object Pose Estimation   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # TODO: Consider extending BubblePCReconstructorDepth with a modified get_imprint methods

    # estimate the contact imprint for each bubble (right and left)
    object_name = 'marker'
    reconstruction_params = load_bubble_reconstruction_params()
    object_params = reconstruction_params[object_name]
    imprint_threshold = object_params['imprint_th']['depth']
    icp_threshold = object_params['icp_th']
    reconstructor = BubblePCReconstructorOfflineDepth(threshold=imprint_threshold, object_name=object_name, estimation_type='icp2d')
    # obtain camera parameters
    camera_info_r = sample['camera_info_r']
    camera_info_l = sample['camera_info_l']
    all_tfs = sample['all_tfs']

    # obtain reference (undeformed) depths
    ref_depth_img_r = sample['undef_depth_l'].squeeze()
    ref_depth_img_l = sample['undef_depth_l'].squeeze()

    # unprocess the imprints (add padding to move them back to the original shape)
    imprint_pred_r = unprocess_bubble_img(imprint_pred_r)
    imprint_pred_l = unprocess_bubble_img(imprint_pred_l)

    deformed_depth_r = ref_depth_img_r - imprint_pred_r # CAREFUL: Imprint is defined as undef_depth_img - def_depth_img
    deformed_depth_l = ref_depth_img_l - imprint_pred_l

    # THIS hacks the ways to obtain data for the reconstructor
    reconstructor.references['left'] = ref_depth_img_l
    reconstructor.references['right'] = ref_depth_img_r
    reconstructor.depth_r = {'img': deformed_depth_r, 'frame':'pico_flexx_right_optical_frame'}
    reconstructor.depth_l = {'img': deformed_depth_l, 'frame':'pico_flexx_left_optical_frame'}
    reconstructor.camera_info['right'] = camera_info_r
    reconstructor.camera_info['left'] = camera_info_l
    # compute transformations from camera frames to grasp frame and transform the
    reconstructor.add_tfs(sample['all_tfs'])
    # estimate pose
    import pdb; pdb.set_trace()
    estimated_pose = reconstructor.estimate_pose(icp_threshold)
