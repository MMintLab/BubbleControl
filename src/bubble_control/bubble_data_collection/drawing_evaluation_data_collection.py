import numpy as np
import torch
import os
from tqdm import tqdm

from bubble_control.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from bubble_control.bubble_learning.models.bubble_dynamics_model import BubbleDynamicsModel
from bubble_control.bubble_model_control.aux.bubble_dynamics_fixed_model import BubbleDynamicsFixedModel
from bubble_control.bubble_learning.models.bubble_linear_dynamics_model import BubbleLinearDynamicsModel
from bubble_control.bubble_learning.models.object_pose_dynamics_model import ObjectPoseDynamicsModel
from victor_hardware_interface_msgs.msg import ControlMode

from bubble_control.bubble_model_control.model_output_object_pose_estimaton import \
    BatchedModelOutputObjectPoseEstimation, End2EndModelOutputObjectPoseEstimation
from bubble_control.bubble_model_control.controllers.bubble_model_mppi_controler import BubbleModelMPPIController
from bubble_control.bubble_envs.bubble_drawing_env import BubbleOneDirectionDrawingEnv

from bubble_control.bubble_model_control.drawing_action_models import drawing_action_model_one_dir, drawing_one_dir_grasp_pose_correction
from bubble_control.bubble_learning.aux.load_model import load_model_version
from bubble_control.aux.drawing_evaluator import DrawingEvaluator
from bubble_control.bubble_model_control.aux.format_observation import format_observation_sample
from bubble_control.bubble_model_control.cost_functions import vertical_tool_cost_function


from bubble_utils.bubble_data_collection.data_collector_base import DataCollectorBase

from mmint_camera_utils.recorders.recording_utils import record_image_color
from mmint_camera_utils.recorders.data_recording_wrappers import ActionSelfSavedWrapper
from arc_utilities.tf2wrapper import TF2Wrapper


class DrawingEvaluationDataCollection(DataCollectorBase):

    def __init__(self, *args, model_name='random', load_version=0, scene_name='drawing_evaluation', imprint_selection='percentile',
                                                     imprint_percentile=0.005,  object_name='marker', debug=False, **kwargs):
        self.scene_name = scene_name
        self.object_name = object_name
        self.num_samples = 100
        self.horizon = 2
        self.init_action = {
            'start_point': np.array([0.55, 0.2]),
            'direction': np.deg2rad(270),
        }
        self.model_name = model_name
        self.load_version = load_version
        self.imprint_selection = imprint_selection
        self.imprint_percentile = imprint_percentile
        self.debug = debug
        self.model_data_path = '/home/mmint/Desktop/drawing_models' # THIS is the path where we expect to load the model. Inside contains tb_logs/{model_name}/version_{version}/....
        self.reference_fc = None
        self.bubble_ref_obs = None
        self.model = self._get_model()
        self.block_downsample_tr = BlockDownSamplingTr(factor_x=7, factor_y=7, reduction='mean', keys_to_tr=['init_imprint'])
        self.ope = self._get_object_pose_estimation()
        self.evaluator = self._get_evaluator()
        self.env = None
        self.controller = None
        super().__init__(*args, **kwargs)
        self.data_save_params = {'save_path': self.data_path, 'scene_name': self.scene_name}

    def _get_legend_column_names(self):
        """
        Return a list containing the column names of the datalegend
        Returns:
        """
        column_names = ['EvaluationFileCode', 'ReferenceFileCode', 'ActionsFileCode', 'ObjectName', 'SceneName', 'ControllerMethod', 'Score', 'NumSteps', 'NumStepsExpected', 'ObservationFileCodes']
        return column_names

    def _get_legend_lines(self, data_params):
        """
        Return a list containing the values to log inot the data legend for the data sample with file code filecode
        Args:
            data_params: <dict> containg parameters of the collected data
        Returns:
        """
        column_names = self._get_legend_column_names()
        lines = [[data_params[cn] for cn in column_names]]
        return lines

    def _collect_data_sample(self, params=None):
        """
        Collect and save data to the designed path in self.data_path
        Args:
            params:
        Returns: <dict> containing the parameters of the collected sample
        """

        self._init_collection_sample()
        if self.bubble_ref_obs is not self.env.bubble_ref_obs:
            # record the reference
            self._record_reference_state()

        # Draw
        num_steps = 40
        num_steps_done = 0
        self.env.do_init_action(self.init_action)
        num_steps_done, obs_fcs, actions = self.draw_steps(num_steps=num_steps)

        fc = self.get_new_filecode()

        actions_self_saved = ActionSelfSavedWrapper(actions, data_params=self.data_save_params)
        actions_self_saved.save_fc(fc)

        # Evaluate
        self.env.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
        self.env.med.home_robot()
        self.env.med.set_robot_conf('zero_conf')
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
            'EvaluationFileCode': fc,
            'ObservationFileCodes': obs_fcs,
            'ReferenceFileCode': self.reference_fc,
            'SceneName': self.scene_name,
            'NumSteps': num_steps_done,
            'ActionsFileCode': fc,
            'NumStepsExpected': num_steps,
            'ControllerMethod': self._get_controller_name(),
            'ObjectName': self.object_name,
            'Score': score,
        }

        return data_params

    def _init_collection_sample(self):
        self.env = self._get_env() # Reset the env every time
        self.controller = self._get_controller() # Reset the controller every time

    def _get_model(self):
        models = [BubbleDynamicsModel, BubbleLinearDynamicsModel, ObjectPoseDynamicsModel]
        model_names = [m.get_name() for m in models]
        if self.model_name in model_names:
            Model = models[model_names.index(self.model_name)]
            model = load_model_version(Model, self.model_data_path, self.load_version)
        elif self.model_name in ['random', 'fixed_model']:
            model = BubbleDynamicsFixedModel() # TODO: Find another way to set the random without using the fixed model.
        else:
            raise AttributeError('Model name provided {} not supported. We currently support {}. We also support "random" and "fixed_model"'.format(self.model_name, model_names))
        print(' \n\n MODEL NAME: {}\n\n'.format(self.model_name))
        model.eval()
        return model

    def _get_object_pose_estimation(self):
        if self.model_name in ['object_pose_dynamics_model']:
            # We do not need to estimate the pose from imprints since the model predicts directly the object pose.
            ope = End2EndModelOutputObjectPoseEstimation()
        else:
            ope = BatchedModelOutputObjectPoseEstimation(object_name='marker', factor_x=7, factor_y=7, method='bilinear',
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
                                           wrap_data=True,
                                           marker_code=self.object_name,
                                           grasp_width_limits=(10, 35))
        return env

    def _get_controller(self):
        if isinstance(self.model, ObjectPoseDynamicsModel):
            grasp_pose_correction = None # get default one which does not correct the pose since it is direclty predicted corrected.
        else:
            grasp_pose_correction = drawing_one_dir_grasp_pose_correction
        controller = BubbleModelMPPIController(self.model, self.env, self.ope, vertical_tool_cost_function,
                                                action_model=drawing_action_model_one_dir,
                                                grasp_pose_correction=grasp_pose_correction,
                                                num_samples=self.num_samples, horizon=self.horizon, noise_sigma=None,
                                                _noise_sigma_value=.3, debug=self.debug)
        return controller

    def _get_controller_name(self):
        if self.model_name == 'random':
            return 'random_action'
        else:
            return '{}_mppi'.format(self.model.name)

    def _get_evaluator(self):
        drawing_evaluator = DrawingEvaluator()
        return drawing_evaluator

    def _get_expected_drawing(self):
        num_points = 1000
        edc_x = self.init_action['start_point'][0] * np.ones((num_points,))
        edc_y = np.linspace(self.init_action['start_point'][1] - 0.55, self.init_action['start_point'][1], num=num_points)
        edc_z = 0.01*np.ones((num_points,))
        expected_drawing_cooridnates = np.stack([edc_x, edc_y, edc_z], axis=-1)
        return expected_drawing_cooridnates

    def draw_steps(self, num_steps):
        obs_fcs = []
        actions = []
        init_obs_sample = self.env.get_observation()
        init_obs_sample.modify_data_params(self.data_save_params)
        fc_init = self.get_new_filecode()
        init_obs_sample.save_fc(fc_init)
        obs_fcs.append(fc_init)
        obs_sample_raw = init_obs_sample.copy()
        step_i = 0
        for step_i in tqdm(range(num_steps)):
            # Downsample the sample
            action, valid_action = self.env.get_action()  # this is a
            obs_sample = self.format_raw_observation(obs_sample_raw)
            if not self.model_name == 'random':
                action = self.controller.control(obs_sample) # it is already an action dictionary
            # print('Action:', action)
            actions.append(action)
            obs_sample_raw, reward, done, info = self.env.step(action)
            fc_i = self.get_new_filecode()
            obs_sample_raw.modify_data_params(self.data_save_params)
            obs_sample_raw.save_fc(fc_i)
            obs_fcs.append(fc_i)
            if self.debug:
                downsampled_obs = self.format_raw_observation(obs_sample_raw)
                self.controller.visualize_prediction(downsampled_obs)
            if done:
                break
        return step_i+1, obs_fcs, actions

    def _record_reference_state(self):
        self.bubble_ref_obs = self.env.bubble_ref_obs
        self.reference_fc = self.get_new_filecode()
        self.env.bubble_ref_obs.modify_data_params(self.data_save_params)
        self.env.bubble_ref_obs.save_fc(self.reference_fc)

    def format_raw_observation(self, obs_sample_raw=None):
        if obs_sample_raw is None:
            obs_sample_raw = self.env.get_observation()
        format_obs = format_observation_sample(obs_sample_raw)
        downsampled_obs = self.block_downsample_tr(format_obs)
        return downsampled_obs
