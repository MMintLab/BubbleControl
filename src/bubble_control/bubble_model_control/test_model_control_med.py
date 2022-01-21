import numpy as np
import torch
import rospy
import pytorch3d.transforms as batched_tr

from bubble_control.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from bubble_control.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from bubble_control.bubble_learning.models.bubble_dynamics_pretrained_ae_model import BubbleDynamicsPretrainedAEModel
from bubble_control.bubble_model_control.aux.bubble_dynamics_fixed_model import BubbleDynamicsFixedModel

from bubble_control.bubble_model_control.model_output_object_pose_estimaton import \
    BatchedModelOutputObjectPoseEstimation
from bubble_control.bubble_model_control.controllers.bubble_model_mppi_controler import BubbleModelMPPIBatchedController
from bubble_control.bubble_envs.bubble_drawing_env import BubbleOneDirectionDrawingEnv
from bubble_utils.bubble_tools.bubble_img_tools import process_bubble_img

from bubble_control.bubble_model_control.drawing_action_models import drawing_action_model_one_dir
from bubble_control.bubble_learning.aux.load_model import load_model_version


def format_observation_sample(obs_sample):
    formatted_obs_sample = {}
    # obs sample should have:
    #           'init_imprint',
    #           'init_wrench',
    #           'init_pos',
    #           'init_quat',
    #           'final_imprint',
    #           'final_wrench',
    #           'final_pos',
    #           'final_quat',
    #           'action',
    #           'undef_depth_r',
    #           'undef_depth_l',
    #           'camera_info_r',
    #           'camera_info_l',
    #           'all_tfs'
    # input input expected keys:
    #           'bubble_camera_info_color_right',
    #           'bubble_camera_info_depth_right',
    #           'bubble_color_img_right',
    #           'bubble_depth_img_right',
    #           'bubble_point_cloud_right',
    #           'bubble_camera_info_color_left',
    #           'bubble_camera_info_depth_left',
    #           'bubble_color_img_left',
    #           'bubble_depth_img_left',
    #           'bubble_point_cloud_left',
    #           'wrench',
    #           'tfs',
    #           'bubble_color_img_right_reference',
    #           'bubble_depth_img_right_reference',
    #           'bubble_point_cloud_right_reference',
    #           'bubble_color_img_left_reference',
    #           'bubble_depth_img_left_reference',
    #           'bubble_point_cloud_left_reference'
    # remap keys ---
    key_map = {
        'tfs': 'all_tfs',
        'bubble_camera_info_depth_left': 'camera_info_l',
        'bubble_camera_info_depth_right': 'camera_info_r',
        'bubble_depth_img_right_reference': 'undef_depth_r',
        'bubble_depth_img_left_reference': 'undef_depth_l',
    }
    for k_old, k_new in key_map.items():
        formatted_obs_sample[k_new] = obs_sample[k_old]
    # add imprints: -------
    init_imprint_r = obs_sample['bubble_depth_img_right_reference'] - obs_sample['bubble_depth_img_right']
    init_imprint_l = obs_sample['bubble_depth_img_left_reference'] - obs_sample['bubble_depth_img_left']
    formatted_obs_sample['init_imprint'] = process_bubble_img(np.stack([init_imprint_r, init_imprint_l], axis=0))[...,0]

    # apply the key_map
    return formatted_obs_sample


if __name__ == '__main__':
    
    rospy.init_node('drawin_model_mmpi_test')

    object_name = 'marker'
    num_samples = 100
    horizon = 2
    random_action = False # Set to False for controls (Mppi)


    # load model:
    # learned model -------------------
    data_name = '/home/mmint/Desktop/drawing_data_one_direction'
    load_version = 0
    Model = BubbleDynamicsPretrainedAEModel
    model = load_model_version(Model, data_name, load_version)
    # Fixed model ----------------------
    # model = BubbleDynamicsFixedModel()


    block_downsample_tr = BlockDownSamplingTr(factor_x=7, factor_y=7, reduction='mean', keys_to_tr=['init_imprint'])

    env = BubbleOneDirectionDrawingEnv(prob_axis=0.08,
                             impedance_mode=False,
                             reactive=False,
                             drawing_area_center=(0.55, 0.),
                             drawing_area_size=(0.15, 0.3),
                             drawing_length_limits=(0.01, 0.02),
                             wrap_data=False,
                             grasp_width_limits=(15, 25))

    # Object Pose Estimation Algorithms:
    # ope = BatchedModelOutputObjectPoseEstimation(object_name=object_name, factor_x=7, factor_y=7, method='bilinear', device=torch.device('cuda'), imprint_selection='threshold') #thresholded imprint esimation
    ope = BatchedModelOutputObjectPoseEstimation(object_name=object_name, factor_x=7, factor_y=7, method='bilinear', device=torch.device('cuda'), imprint_selection='percentile', imprint_percentile=0.005) #percentile


    def test_cost_function(estimated_poses, states, actions):
        # Only position ----------------------------------------
        # goal_xyz = np.zeros(3)
        # estimated_xyz = estimated_poses[:, :3]
        # cost = np.linalg.norm(estimated_xyz-goal_xyz, axis=1)

        # Only orientation, using model points ------------
        # tool axis is z, so we want tool frame z axis to be aligned with the world z axis
        estimated_pos = estimated_poses[:, :3] # (x, y, z)
        estimated_q = estimated_poses[:, 3:] # (qx,qy,qz,qw)
        estimated_qwxyz = torch.index_select(estimated_q, dim=-1, index=torch.LongTensor([3, 0, 1, 2]))# (qw, qx,qy,qz)
        estimated_R = batched_tr.quaternion_to_matrix(estimated_qwxyz) # careful! batched_tr quat is [qw,qx,qy,qz], we work as [qx,qy,qz,qw]
        z_axis = torch.tensor([0., 0, 1.]).unsqueeze(0).repeat_interleave(estimated_R.shape[0], dim=0).float()
        tool_z_axis_wf = torch.einsum('kij,kj->ki', estimated_R, z_axis)
        ori_cost = 1-torch.abs(torch.einsum('ki,ki->k', z_axis, tool_z_axis_wf)) # TO MINIMIZE (perfect case tool axis parallel to z axis, aka dot(z_axis, tool_z_axis)=1)
        ori_cost = torch.nan_to_num(ori_cost, nan=10.0)
        is_nan_action = torch.any(torch.isnan(actions), dim=1)
        is_nan_pose = torch.any(torch.isnan(estimated_pos), dim=1)
        cost = ori_cost + 10*is_nan_action + 10 * is_nan_pose
        return cost

    controller = BubbleModelMPPIBatchedController(model, env, ope, test_cost_function, action_model=drawing_action_model_one_dir, num_samples=num_samples, horizon=horizon, noise_sigma=None, _noise_sigma_value=1.)


    # +++++++++++++++++++++++
    # DRAWING FUNCTION
    def draw_steps(num_steps):
        init_obs_sample = env.get_observation()
        obs_sample_raw = init_obs_sample.copy()
        for i in range(num_steps):
            # Downsample the sample
            action, valid_action = env.get_action()  # this is a
            obs_sample = format_observation_sample(obs_sample_raw)
            obs_sample = block_downsample_tr(obs_sample)

            if not random_action:
                action_raw = controller.control(obs_sample).detach().cpu().numpy()
                print(action_raw)
                if np.isnan(action_raw).any():
                    print('Nan Value --- {}'.format(action_raw))
                    break
                for i, (k, v) in enumerate(action.items()):
                    action[k] = action_raw[i]
            print('Action:', action)
            obs_sample_raw, reward, done, info = env.step(action)
            if done:
                return i

    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Control -- STRAIGTH LINE  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    init_action = {
        'start_point': np.array([0.55, 0.2]),
        'direction': np.deg2rad(270),
    }
    env.do_init_action(init_action)
    draw_steps(40)

    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Control -- Triangle  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # init_action = {
    #     'start_point': np.array([0.50, 0.2]),
    #     'direction': np.deg2rad(270),
    # }
    # env.do_init_action(init_action)
    # draw_steps(8)
    #
    # env.change_drawing_direction(np.deg2rad(0))
    # draw_steps(8)
    # env.change_drawing_direction(np.deg2rad(90))
    # draw_steps(8)
    # env.change_drawing_direction(np.deg2rad(180))
    # draw_steps(8)

