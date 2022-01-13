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

from bubble_control.bubble_model_control.model_output_object_pose_estimaton import ModelOutputObjectPoseEstimation, BatchedModelOutputObjectPoseEstimation
from bubble_control.bubble_model_control.bubble_model_controler import BubbleModelMPPIController, BubbleModelMPPIBatchedController
from bubble_control.bubble_envs.bubble_drawing_env import BubbleOneDirectionDrawingEnv


def load_model_version(Model, data_name, load_version):
    model_name = Model.get_name()
    version_chkp_path = os.path.join(data_name, 'tb_logs', '{}'.format(model_name),
                                     'version_{}'.format(load_version), 'checkpoints')
    checkpoints_fs = [f for f in os.listdir(version_chkp_path) if
                      os.path.isfile(os.path.join(version_chkp_path, f))]
    checkpoint_path = os.path.join(version_chkp_path, checkpoints_fs[0])

    model = Model.load_from_checkpoint(checkpoint_path)
    return model


def test_cost_function(estimated_poses, states, actions):
    goal_xyz = np.zeros(3)
    estimated_xyz = estimated_poses[:, :3]
    cost = np.linalg.norm(estimated_xyz-goal_xyz, axis=1)
    return cost


if __name__ == '__main__':
    
    rospy.init_node('drawin_model_mmpi_test')
    
    data_name = '/home/mik/Desktop/drawing_data_one_direction'
    load_version = 0
    object_name = 'marker'
    Model = BubbleDynamicsPretrainedAEModel

    num_samples = 150
    horizon = 3

    block_downsample_tr = BlockDownSamplingTr(factor_x=7, factor_y=7, reduction='mean', keys_to_tr=['init_imprint'])

    # load model:
    model = load_model_version(Model, data_name, load_version)

    env = BubbleOneDirectionDrawingEnv(prob_axis=0.08,
                             impedance_mode=False,
                             reactive=False,
                             drawing_area_center=(0.55, 0.),
                             drawing_area_size=(0.15, 0.3),
                             drawing_length_limits=(0.01, 0.02),
                             wrap_data=True,
                             grasp_width_limits=(15,25))

    ope = BatchedModelOutputObjectPoseEstimation(object_name=object_name, factor_x=7, factor_y=7, method='bilinear')
    controller = BubbleModelMPPIBatchedController(model, env, ope, test_cost_function, num_samples=num_samples, lambda_=1., horizon=horizon, noise_sigma=None)

    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Control   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    env.initialize()
    init_obs_sample = env.get_observation()
    obs_sample = init_obs_sample.copy()
    for i in range(10):
        # Downsample the sample
        action, valid_action = env.get_action()
        action_raw = controller.control(obs_sample)
        for i, (k, v) in enumerate(action.items()):
            action[k] = action_raw[i]
        obs_sample, reward, done, info = env.step(action)







