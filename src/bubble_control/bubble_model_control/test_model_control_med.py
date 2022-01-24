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
from bubble_control.aux.drawing_evaluator import DrawingEvaluator
from bubble_control.bubble_model_control.aux.format_observation import format_observation_sample
from bubble_control.bubble_model_control.cost_functions import vertical_tool_cost_function




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


    controller = BubbleModelMPPIBatchedController(model, env, ope, vertical_tool_cost_function, action_model=drawing_action_model_one_dir, num_samples=num_samples, horizon=horizon, noise_sigma=None, _noise_sigma_value=.3)


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


    # EVALUATOR -----------
    drawing_evaluator = DrawingEvaluator()



    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Control -- STRAIGTH LINE  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    init_action = {
        'start_point': np.array([0.55, 0.2]),
        'direction': np.deg2rad(270),
    }
    # env.do_init_action(init_action)
    # draw_steps(40)
    env.med.home_robot()
    env.med.set_robot_conf('zero_conf')


    num_points = 1000
    edc_x = init_action['start_point'][0]*np.ones((num_points,))
    edc_y = np.linspace(init_action['start_point'][1]-0.4,init_action['start_point'][1], num=num_points)
    edc_z = np.zeros((num_points,))
    expected_drawing_cooridnates = np.stack([edc_x, edc_y, edc_z], axis=-1)
    score, _, _ = drawing_evaluator.evaluate(expected_drawing_cooridnates, frame='med_base', save_path='/home/mmint/Desktop/drawing_evaluation/test_model_control_med')
    print('SCORE: ', score)

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

