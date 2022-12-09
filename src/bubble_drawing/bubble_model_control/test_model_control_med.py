import numpy as np
import torch
import rospy
import tf.transformations as tr

from bubble_drawing.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from bubble_drawing.bubble_learning.models.old.bubble_dynamics_pretrained_ae_model import BubbleDynamicsPretrainedAEModel
from bubble_drawing.bubble_model_control.aux.bubble_dynamics_fixed_model import BubbleDynamicsFixedModel

from bubble_drawing.bubble_model_control.model_output_object_pose_estimaton import BatchedModelOutputObjectPoseEstimation
from bubble_drawing.bubble_model_control.controllers.bubble_model_mppi_controler import BubbleModelMPPIBatchedController
from bubble_drawing.bubble_envs.bubble_drawing_env import BubbleOneDirectionDrawingEnv

from bubble_drawing.bubble_model_control.drawing_action_models import drawing_action_model_one_dir
from bubble_drawing.bubble_learning.aux.load_model import load_model_version
from bubble_drawing.bubble_model_control.aux.format_observation import format_observation_sample
from bubble_drawing.bubble_model_control.cost_functions import vertical_tool_cost_function
from victor_hardware_interface_msgs.msg import ControlMode




if __name__ == '__main__':
    
    rospy.init_node('drawin_model_mmpi_test')

    object_name = 'marker'
    num_samples = 100
    horizon = 2
    random_action = False # Set to False for controls (Mppi)
    fixed_model = False # set false for learned model


    # load model:
    # learned model -------------------
    data_name = '/home/mmint/Desktop/drawing_data_one_direction'
    load_version = 0
    Model = BubbleDynamicsPretrainedAEModel
    model = load_model_version(Model, data_name, load_version)
    # Fixed model ----------------------
    if fixed_model:
        model = BubbleDynamicsFixedModel()


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


    controller = BubbleModelMPPIBatchedController(model, env, ope, vertical_tool_cost_function, action_model=drawing_action_model_one_dir, num_samples=num_samples, horizon=horizon, noise_sigma=None, _noise_sigma_value=.3)


    # +++++++++++++++++++++++
    # DRAWING FUNCTION
    def draw_steps(num_steps):
        init_obs_sample = env.get_observation()
        obs_sample_raw = init_obs_sample.copy()
        for i in range(num_steps):
            action, valid_action = env.get_action()  # this is to get the action container to fill and therefore get the correct format.
            obs_sample = format_observation_sample(obs_sample_raw)
            obs_sample = block_downsample_tr(obs_sample) # Downsample the sample
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


    def draw_to_point(point_x, point_y, max_num_steps=40):
        completed = False
        goal_xy = np.array([point_x, point_y])
        # Check if we are drawing in the right direction, if not reorient the drawing by pivoting along
        current_drawing_direction = env._get_current_drawing_direction()
        unit_drawing_vect = np.array([np.cos(current_drawing_direction), np.sin(current_drawing_direction)])
        # current_planar_pos = env.med.get_plane_pose()[:3] # grasp pose
        current_planar_pos = env.get_contact_point()[:3] # contact point pose
        move_vect = goal_xy - current_planar_pos[:2]
        # get the angle between the current direction and the desired one
        R = tr.quaternion_matrix(tr.quaternion_about_axis(angle=current_drawing_direction, axis=(0,0,1)))[:2,:2]
        move_vect_df = R.T @ move_vect # move_vect expressed in the drawing frame
        diff_angle = np.arctan2(move_vect_df[1], move_vect_df[0])
        if abs(diff_angle) > np.deg2rad(3):
            # Rotate to reorient
            env.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
            env.med.rotation_along_axis_point_angle(axis=np.array([0, 0, 1]), point=current_planar_pos, angle=diff_angle, num_steps=1)
            env._set_cartesian_impedance()

        init_obs_sample = env.get_observation()
        obs_sample_raw = init_obs_sample.copy()
        for i in range(max_num_steps):
            action, valid_action = env.get_action()  # this is to get the action container to fill and therefore get the correct format.
            obs_sample = format_observation_sample(obs_sample_raw)
            obs_sample = block_downsample_tr(obs_sample)  # Downsample the sample
            if not random_action:
                action_raw = controller.control(obs_sample).detach().cpu().numpy()
                if np.isnan(action_raw).any():
                    print('Nan Value --- {}'.format(action_raw))
                    break
                for i, (k, v) in enumerate(action.items()):
                    action[k] = action_raw[i]
            print('Action:', action)
            obs_sample_raw, reward, done, info = env.step(action)
            if done:
                break
            # check if we have reached the goal (or passed it)
            current_drawing_direction = env._get_current_drawing_direction()
            # current_planar_pos = env.med.get_plane_pose()[:3]  # grasp pose
            current_planar_pos = env.get_contact_point()[:3]  # contact point pose
            move_vect = goal_xy - current_planar_pos[:2]
            unit_drawing_vect = np.array([np.cos(current_drawing_direction), np.sin(current_drawing_direction)])
            goal_dist = np.linalg.norm(move_vect)
            unit_move_vect = move_vect/goal_dist
            signed_goal_dist = np.sign(np.dot(unit_drawing_vect, unit_move_vect)) * goal_dist
            if signed_goal_dist <= 0:
                print('Goal Reached')
                completed = True
                break
        return completed


    def draw_collection_points(points, num_iters=1):
        init_point = points[0]
        moving_delta_init = points[1]-points[0]
        init_dir = np.arctan2(moving_delta_init[1], moving_delta_init[0])%(2*np.pi)

        init_action = {
            'start_point': init_point,
            'direction': init_dir,
        }
        env.do_init_action(init_action)
        for iter in range(num_iters):
            for i, goal_point in enumerate(points[1:]):
                completed = draw_to_point(goal_point[0], goal_point[1])
                if not completed:
                    break
        print('DONE')
        env.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
        env.med.home_robot()



    # EVALUATOR -----------
    # drawing_evaluator = DrawingEvaluator()



    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Control -- STRAIGTH LINE  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    init_action = {
        'start_point': np.array([0.55, 0.2]),
        'direction': np.deg2rad(270),
    }
    env.do_init_action(init_action)
    draw_steps(40)
    env.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
    env.med.home_robot()
    env.med.set_robot_conf('zero_conf')


    # num_points = 1000
    # edc_x = init_action['start_point'][0]*np.ones((num_points,))
    # edc_y = np.linspace(init_action['start_point'][1]-0.4,init_action['start_point'][1], num=num_points)
    # edc_z = np.zeros((num_points,))
    # expected_drawing_cooridnates = np.stack([edc_x, edc_y, edc_z], axis=-1)
    # score, _, _ = drawing_evaluator.evaluate(expected_drawing_cooridnates, frame='med_base', save_path='/home/mmint/Desktop/drawing_evaluation/test_model_control_med')
    # print('SCORE: ', score)


    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Control -- Square Open Loop  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # init_action = {
    #     'start_point': np.array([0.50, 0.2]),
    #     'direction': np.deg2rad(270),
    # }
    # env.do_init_action(init_action)
    # for i in range(4):
    #     draw_steps(8)
    #     env.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
    #     env.med.rotation_along_axis_point_angle(np.array([0, 0, 1]), np.pi / 180 * 90, num_steps=1)
    #     env._set_cartesian_impedance()
    #
    # # print('DONE')

    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Control -- Square Closed Loop  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # square_points = np.array([
    #     [0.50, 0.2],
    #     [0.5, 0.1],
    #     [0.6, 0.1],
    #     [0.6, 0.2],
    #     [0.50, 0.2]
    # ])
    # draw_collection_points(square_points)


    # #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Control -- Draw sideways  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # init_action = {
    #     'start_point': np.array([0.55, 0.2]),
    #     'direction': np.deg2rad(270),
    # }
    # env.do_init_action(init_action)
    # env.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
    # env.med.rotation_along_axis_point_angle(np.array([0, 0, 1]), np.pi / 180 * 90)
    # env._set_cartesian_impedance()
    # draw_steps(40)
    # print('DONE')


