import gym
import abc

import numpy as np
import rospy
from collections import OrderedDict
import gym
import copy

from mmint_camera_utils.recorders.data_recording_wrappers import DataSelfSavedWrapper
from bubble_control.bubble_drawer.bubble_drawer import BubbleDrawer
from bubble_control.aux.action_sapces import ConstantSpace, AxisBiasedDirectionSpace
from bubble_control.bubble_envs.base_env import BubbleBaseEnv



class BubbleDrawingEnv(BubbleBaseEnv):

    def __init__(self, *args, impedance_mode=False, reactive=False, force_threshold=5., prob_axis=0.08,
                 drawing_area_center=(0.55, 0.), drawing_area_size=(.15, .15), drawing_length_limits=(0.01, 0.15),
                 grasp_width_limits=(15,25), **kwargs):
        self.impedance_mode = impedance_mode
        self.reactive = reactive
        self.force_threshold = force_threshold
        self.prob_axis = prob_axis
        self.drawing_area_center = drawing_area_center
        self.drawing_area_size = drawing_area_size
        self.drawing_length_limits = drawing_length_limits
        self.grasp_width_limits = grasp_width_limits
        self.previous_end_point = None
        self.previous_draw_height = None
        self.drawing_init = False
        self.bubble_ref_obs = None
        self.init_action_space = self._get_init_action_space()
        super().__init__(*args, **kwargs)
        self.reset()

    @classmethod
    def get_name(cls):
        return 'bubble_drawing_env'

    def reset(self):
        self.med.set_grasp_pose()
        _ = input('Press enter to open the gripper and calibrate the bubbles')
        self.med.open_gripper()
        # Calibrate
        self.bubble_ref_obs = self._get_bubble_observation()
        _ = input('Press enter to close the gripper')
        self.med.set_grasping_force(5.0)
        self.med.gripper.move(25.0)
        self.med.grasp(20.0, 30.0)
        rospy.sleep(2.0)
        print('Calibration is done')
        self.med.home_robot()
        super().reset()

    def initialize(self):
        init_action = self.init_action_space.sample()
        start_point_i = init_action['start_point']
        if self.drawing_init:
            self.med._end_raise()
        draw_height = self.med._init_drawing(start_point_i)
        self.previous_draw_height = copy.deepcopy(draw_height)


    def _get_med(self):
        med = BubbleDrawer(object_topic='estimated_object',
                           wrench_topic='/med/wrench',
                           force_threshold=self.force_threshold,
                           reactive=self.reactive,
                           impedance_mode=self.impedance_mode)
        med.connect()
        return med

    def _get_action_space(self):
        action_space_dict = OrderedDict()
        action_space_dict['direction'] = AxisBiasedDirectionSpace(prob_axis=self.prob_axis)
        action_space_dict['length'] = gym.spaces.Box(low=self.drawing_length_limits[0], high=self.drawing_length_limits[1], shape=())
        action_space_dict['grasp_width'] = gym.spaces.Box(low=self.grasp_width_limits[0], high=self.grasp_width_limits[1], shape=())

        action_space = gym.spaces.Dict(action_space_dict)
        return action_space

    def _get_init_action_space(self):
        drawing_area_center_point = np.asarray(self.drawing_area_center)
        drawing_area_size = np.asarray(self.drawing_area_size)

        action_space_dict = OrderedDict()
        action_space_dict['start_point'] = gym.spaces.Box(drawing_area_center_point - drawing_area_size,
                                          drawing_area_center_point + drawing_area_size, (2,), dtype=np.float64) # random uniform
        action_space = gym.spaces.Dict(action_space_dict)
        return action_space

    def _check_valid_action(self, action):
        direction_i = action['direction']
        length_i = action['length']
        grasp_width_i = action['grasp_width']
        drawing_area_center_point = np.asarray(self.drawing_area_center)
        drawing_area_size = np.asarray(self.drawing_area_size)
        self.med.gripper.move(grasp_width_i, 10.0)
        current_point_i = self._get_robot_plane_position()
        end_point_i = current_point_i + length_i * np.array([np.cos(direction_i), np.sin(direction_i)])
        # Check if the end_point will be whitin the limits:
        valid_action = np.all(end_point_i<=drawing_area_center_point + drawing_area_size) and np.all(end_point_i >= drawing_area_center_point - drawing_area_size)
        return valid_action

    def _add_bubble_reference_to_observation(self, obs):
        keys_to_include = ['color_img', 'depth_img', 'point_cloud']
        for k, v in self.bubble_ref_obs.items():
            for kti in keys_to_include:
                if kti in k:
                    # do not add the saving method:
                    if isinstance(v, DataSelfSavedWrapper):
                        obs['{}_reference'.format(k)] = v.data # unwrap the data so it will not be saved with the observation. This avoid overriting reference and current state. Reference will be saved apart.
                    else:
                        obs['{}_reference'.format(k)] = v
        return obs

    def _get_observation(self):
        obs = {}
        bubble_obs = self._get_bubble_observation()
        obs.update(bubble_obs)
        obs['wrench'] = self._get_wrench()
        obs['tfs'] = self._get_tfs()
        # add the reference state
        obs = self._add_bubble_reference_to_observation(obs)
        return obs

    def _get_robot_plane_position(self):
        plane_pose = self.med.tf2_listener.get_transform(parent=self.med.drawing_frame, child='grasp_frame')
        plane_pos_xy = plane_pose[:2,3]
        return plane_pos_xy

    def _do_action(self, action):
        direction_i = action['direction']
        length_i = action['length']
        grasp_width_i = action['grasp_width']
        self.med.gripper.move(grasp_width_i, 10.0)
        current_point_i = self._get_robot_plane_position()
        end_point_i = current_point_i + length_i * np.array([np.cos(direction_i), np.sin(direction_i)])
        self.med._draw_to_point(end_point_i, self.previous_draw_height)
