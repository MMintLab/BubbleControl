import numpy as np
import os
import sys
import torch
import copy
import rospy
import tf2_ros as tf
import tf.transformations as tr
from geometry_msgs.msg import TransformStamped

from bubble_control.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from bubble_control.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from bubble_control.bubble_learning.aux.img_trs.block_upsampling_tr import BlockUpSamplingTr
from bubble_control.bubble_learning.models.bubble_dynamics_pretrained_ae_model import BubbleDynamicsPretrainedAEModel

from bubble_control.bubble_model_control.model_output_object_pose_estimaton import ModelOutputObjectPoseEstimation, ModelDownsampledOutputObjectPoseEstimation
from bubble_control.bubble_model_control.bubble_model_controler import BubbleModelMPPIController



def load_model_version(Model, data_name, load_version):
    model_name = Model.get_name()
    version_chkp_path = os.path.join(data_name, 'tb_logs', '{}'.format(model_name),
                                     'version_{}'.format(load_version), 'checkpoints')
    checkpoints_fs = [f for f in os.listdir(version_chkp_path) if
                      os.path.isfile(os.path.join(version_chkp_path, f))]
    checkpoint_path = os.path.join(version_chkp_path, checkpoints_fs[0])

    model = Model.load_from_checkpoint(checkpoint_path)
    return model


if __name__ == '__main__':

    data_name = '/home/mik/Desktop/drawing_data_cartesian'
    dataset = BubbleDrawingDataset(data_name=data_name, wrench_frame='med_base', tf_frame='grasp_frame')
    Model = BubbleDynamicsPretrainedAEModel

    load_version = 11

    block_downsample_tr = BlockDownSamplingTr(factor_x=7, factor_y=7, reduction='max', keys_to_tr=['init_imprint'])

    # load model:
    model = load_model_version(Model, data_name, load_version)

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
    
    sample_out = copy.deepcopy(sample)
    sample_out['next_imprint'] = next_imprint


    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Object Pose Estimation   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # estimate the contact imprint for each bubble (right and left)
    object_name = 'marker'
    ope = ModelDownsampledOutputObjectPoseEstimation(object_name=object_name, factor_x=7, factor_y=7, method='bilinear')

    # estimate pose
    estimated_pose = ope.estimate_pose(sample_out)


    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   TEST Control   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def test_cost_function(estimated_poses, states, actions):
        goal_xyz = np.zeros(3)
        estimated_xyz = estimated_poses[:, :3]
        cost = np.linalg.norm(estimated_xyz-goal_xyz, axis=1)
        return cost
    import pdb; pdb.set_trace()
    controller = BubbleModelMPPIController(model, ope, test_cost_function)


