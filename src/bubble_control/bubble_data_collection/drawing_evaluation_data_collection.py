import numpy as np
import torch
import os
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


from bubble_utils.bubble_data_collection.data_collector_base import DataCollectorBase

from mmint_camera_utils.recorders.recording_utils import save_image, record_image_color



class DrawingEvaluationDataCollection(DataCollectorBase):

    def __init__(self, *args, scene_name='drawing_evaluation', random_action=False, fixed_model=False, imprint_selection='percentile',
                                                     imprint_percentile=0.005, **kwargs):
        self.scene_name = scene_name
        self.object_name = 'marker'
        self.num_samples = 100
        self.horizon = 2
        self.init_action = {
            'start_point': np.array([0.55, 0.2]),
            'direction': np.deg2rad(270),
        }
        self.fixed_model = fixed_model
        self.random_action = random_action
        self.imprint_selection = imprint_selection
        self.imprint_percentile = imprint_percentile
        self.data_name = '/home/mmint/Desktop/drawing_data_one_direction'
        self.load_version = 0
        self.model = self._get_model()
        self.block_downsample_tr = BlockDownSamplingTr(factor_x=7, factor_y=7, reduction='mean', keys_to_tr=['init_imprint'])
        self.ope = self._get_object_pose_estimation()
        self.evaluator = self._get_evaluator()
        self.env = None
        self.controller = None
        super().__init__(*args, **kwargs)


    def _get_legend_column_names(self):
        """
        Return a list containing the column names of the datalegend
        Returns:
        """
        column_names = ['FileCode', 'SceneName', 'ControllerMethod', 'Score', 'NumSteps', 'NumStepsExpected']
        return column_names

    def _get_legend_lines(self, data_params):
        """
        Return a list containing the values to log inot the data legend for the data sample with file code filecode
        Args:
            data_params: <dict> containg parameters of the collected data
        Returns:
        """
        column_names = self._get_legend_column_names()
        lines = np.array([data_params[cn] for cn in column_names], dtype=object).T
        return lines

    def _collect_data_sample(self, params=None):
        """
        Collect and save data to the designed path in self.data_path
        Args:
            params:
        Returns: <dict> containing the parameters of the collected sample
        """
        fc = self.get_new_filecode()
        self._init_collection_sample()

        # Draw
        num_steps = 40
        self.env.do_init_action(self.init_action)
        num_steps_done = self.draw_steps(num_steps=num_steps)

        # Evaluate
        self.env.med.home_robot()
        self.env.med.set_execute()
        expected_drawing_cooridnates = self._get_expected_drawing()
        score, actual_drawing, expected_drawing = self.evaluator.evaluate(expected_drawing_cooridnates,
                                                                          frame='med_base',
                                                                          save_path=os.path.join(self.data_path, 'evaluation_files', '{:06d}'.format(fc)))
        print('FC {} SCORE: {}'.format(fc, score))
        # Save the actual_drawing and the expected drawing
        record_image_color(img=actual_drawing, save_path=self.data_path, scene_name=self.scene_name, camera_name='measured_drawing', fc=fc, save_as_numpy=True)
        record_image_color(img=expected_drawing, save_path=self.data_path, scene_name=self.scene_name, camera_name='expected_drawing', fc=fc, save_as_numpy=True)

        # pack the score and other significant data (num_steps, ...
        data_params = {
            'FileCode': fc,
            'SceneName': self.scene_name,
            'NumSteps': num_steps_done,
            'NumStepsExpected': num_steps,
            'ControllerMethod': self._get_controller_name(),
            'Score': score,
        }

        return data_params

    def _init_collection_sample(self):
        self.env = self._get_env() # Reset the env every time
        self.controller = self._get_controller() # Reset the controller every time

    def _get_model(self):
        if not self.fixed_model:
            Model = BubbleDynamicsPretrainedAEModel
            model = load_model_version(Model, self.data_name, self.load_version)
        else:
            model = BubbleDynamicsFixedModel()

        return model

    def _get_object_pose_estimation(self):
        ope = BatchedModelOutputObjectPoseEstimation(object_name=self.object_name, factor_x=7, factor_y=7, method='bilinear',
                                                     device=torch.device('cuda'), imprint_selection=self.imprint_selection,
                                                     imprint_percentile=self.imprint_percentile)  # percentile
        return ope

    def _get_env(self):
        env = BubbleOneDirectionDrawingEnv(prob_axis=0.08,
                                           impedance_mode=False,
                                           reactive=False,
                                           drawing_area_center=(0.55, 0.),
                                           drawing_area_size=(0.15, 0.3),
                                           drawing_length_limits=(0.01, 0.02),
                                           wrap_data=False,
                                           grasp_width_limits=(15, 25))
        return env

    def _get_controller(self):
        controller = BubbleModelMPPIBatchedController(self.model, self.env, self.ope, vertical_tool_cost_function,
                                                      action_model=drawing_action_model_one_dir,
                                                      num_samples=self.num_samples, horizon=self.horizon, noise_sigma=None,
                                                      _noise_sigma_value=.3)
        return controller

    def _get_controller_name(self):
        if self.random_action:
            return 'random_action'
        else:
            return '{}_mppi'.format(self.model.name)

    def _get_evaluator(self):
        drawing_evaluator = DrawingEvaluator()
        return drawing_evaluator

    def _get_expected_drawing(self):
        num_points = 1000
        edc_x = self.init_action['start_point'][0] * np.ones((num_points,))
        edc_y = np.linspace(self.init_action['start_point'][1] - 0.4, self.init_action['start_point'][1], num=num_points)
        edc_z = np.zeros((num_points,))
        expected_drawing_cooridnates = np.stack([edc_x, edc_y, edc_z], axis=-1)
        return expected_drawing_cooridnates

    def draw_steps(self, num_steps):
        init_obs_sample = self.env.get_observation()
        obs_sample_raw = init_obs_sample.copy()
        for i in range(num_steps):
            # Downsample the sample
            action, valid_action = self.env.get_action()  # this is a
            obs_sample = format_observation_sample(obs_sample_raw)
            obs_sample = self.block_downsample_tr(obs_sample)

            if not self.random_action:
                action_raw = self.controller.control(obs_sample).detach().cpu().numpy()
                print(action_raw)
                if np.isnan(action_raw).any():
                    print('Nan Value --- {}'.format(action_raw))
                    break
                for i, (k, v) in enumerate(action.items()):
                    action[k] = action_raw[i]
            print('Action:', action)
            obs_sample_raw, reward, done, info = self.env.step(action)
            if done:
                return i
        return num_steps