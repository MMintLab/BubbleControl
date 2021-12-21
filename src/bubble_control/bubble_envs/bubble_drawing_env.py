import gym
import abc
from abc import abstractmethod

import numpy as np
import rospy
from collections import OrderedDict
import gym
import copy
import tf.transformations as tr

from geometry_msgs.msg import Pose, Point, Quaternion

from mmint_camera_utils.recorders.data_recording_wrappers import DataSelfSavedWrapper
from bubble_control.bubble_drawer.bubble_drawer import BubbleDrawer
from bubble_control.aux.action_sapces import ConstantSpace, AxisBiasedDirectionSpace
from bubble_control.bubble_envs.base_env import BubbleBaseEnv
from victor_hardware_interface_msgs.msg import ControlMode
from victor_hardware_interface.victor_utils import get_cartesian_impedance_params, send_new_control_mode




class BubbleDrawingBaseEnv(BubbleBaseEnv):

    def __init__(self, *args, impedance_mode=False, reactive=False, force_threshold=5., prob_axis=0.08,
                 drawing_area_center=(0.55, 0.), drawing_area_size=(.15, .15), drawing_length_limits=(0.01, 0.15),
                 grasp_width_limits=(15, 25), **kwargs):
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
        self.init_action = None
        self.init_action_space = self._get_init_action_space()
        super().__init__(*args, **kwargs)
        self.reset()

    @classmethod
    def get_name(cls):
        return 'bubble_drawing_base_env'

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

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def _get_init_action_space(self):
        pass

    def _get_med(self):
        med = BubbleDrawer(object_topic='estimated_object',
                           wrench_topic='/med/wrench',
                           force_threshold=self.force_threshold,
                           reactive=self.reactive,
                           impedance_mode=self.impedance_mode)
        med.connect()
        return med

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

    def _get_observation_space(self):
        return None

    def _get_robot_plane_position(self):
        plane_pose = self.med.get_plane_pose()
        plane_pos_xy = plane_pose[:2]
        return plane_pos_xy


class BubbleCartesianDrawingEnv(BubbleDrawingBaseEnv):

    def initialize(self):
        self.init_action = self.init_action_space.sample()
        start_point_i = self.init_action['start_point']
        if self.drawing_init:
            self.med._end_raise()
        draw_height = self.med._init_drawing(start_point_i)
        self.previous_draw_height = copy.deepcopy(draw_height)
        self.drawing_init = True

    @classmethod
    def get_name(cls):
        return 'bubble_cartesian_drawing_env'

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

    def is_action_valid(self, action):
        direction_i = action['direction']
        length_i = action['length']
        drawing_area_center_point = np.asarray(self.drawing_area_center)
        drawing_area_size = np.asarray(self.drawing_area_size)
        current_point_i = self._get_robot_plane_position()
        end_point_i = current_point_i + length_i * np.array([np.cos(direction_i), np.sin(direction_i)])
        # Check if the end_point will be whitin the limits:
        valid_action = np.all(end_point_i<=drawing_area_center_point + drawing_area_size) and np.all(end_point_i >= drawing_area_center_point - drawing_area_size)
        return valid_action

    def _do_action(self, action):
        direction_i = action['direction']
        length_i = action['length']
        grasp_width_i = action['grasp_width']
        self.med.gripper.move(grasp_width_i, 10.0)
        current_point_i = self._get_robot_plane_position()
        end_point_i = current_point_i + length_i * np.array([np.cos(direction_i), np.sin(direction_i)])
        planning_result = self.med._draw_to_point(end_point_i, self.previous_draw_height)
        action_feedback = {
            'planning_success': planning_result.planning_result.success,
            'execution_success': planning_result.execution_result.success,
        }
        return action_feedback


class BubbleOneDirectionDrawingEnv(BubbleDrawingBaseEnv):

    def __init__(self, *args, rotation_limits=(-np.pi*5/180, np.pi*5/180), **kwargs):
        self.rotation_limits = rotation_limits
        super().__init__(*args, **kwargs)

    def initialize(self):
        self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
        self.med.home_robot()
        self.init_action = self.init_action_space.sample()
        start_point_i = self.init_action['start_point']
        drawing_direction = self.init_action['direction']
        if self.drawing_init:
            self.med._end_raise()
        # set rotation
        # self.med.rotation_along_axis_point_angle(axis=(0,0,1), angle=drawing_direction)
        rot_quat = tr.quaternion_about_axis(angle=drawing_direction, axis=(0,0,1))# rotation about z
        draw_quat = tr.quaternion_multiply(rot_quat, self.med.draw_quat)
        draw_height = self.med._init_drawing(start_point_i, draw_quat=draw_quat)
        # SET Cartesian Impedance
        cartesian_impedance_params = get_cartesian_impedance_params(velocity=0.25*40) # we multiply by 40 because in get_control_mode they do the same...
        cartesian_impedance_params.cartesian_impedance_params.cartesian_stiffness.z = 2000.0 # by default is 5000
        send_new_control_mode(arm='med', msg=cartesian_impedance_params)
        # self.med.set_control_mode(control_mode=ControlMode.CARTESIAN_IMPEDANCE, vel=0.25)
        self.previous_draw_height = copy.deepcopy(draw_height)
        self.drawing_init = True

    @classmethod
    def get_name(cls):
        return 'bubble_one_direction_drawing_env'

    def _get_action_space(self):
        action_space_dict = OrderedDict()
        action_space_dict['rotation'] = gym.spaces.Box(low=self.rotation_limits[0], high=self.rotation_limits[1], shape=())
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
        action_space_dict['direction'] = AxisBiasedDirectionSpace(prob_axis=self.prob_axis)
        action_space = gym.spaces.Dict(action_space_dict)
        return action_space

    def is_action_valid(self, action):
        direction_i = self.init_action['direction']
        length_i = action['length']
        drawing_area_center_point = np.asarray(self.drawing_area_center)
        drawing_area_size = np.asarray(self.drawing_area_size)
        current_point_i = self._get_robot_plane_position()
        end_point_i = current_point_i + length_i * np.array([np.cos(direction_i), np.sin(direction_i)])
        # Check if the end_point will be whitin the limits:
        valid_action = np.all(end_point_i <= drawing_area_center_point + drawing_area_size) and np.all(end_point_i >= drawing_area_center_point - drawing_area_size)
        return valid_action

    def _is_done(self, observation, a):
        # TODO: Use observation values instead of the current measures!
        current_plane_pose = self.med.get_plane_pose()
        draw_h_limit = self.med.draw_height_limit
        drawing_area_center_point = np.asarray(self.drawing_area_center)
        drawing_area_size = np.asarray(self.drawing_area_size)
        current_h = current_plane_pose[2]
        current_xy = current_plane_pose[:2]
        is_lower_h = current_h < draw_h_limit
        is_out_of_region = not (np.all(current_xy <= drawing_area_center_point + drawing_area_size) and np.all(current_xy >= drawing_area_center_point - drawing_area_size))
        done = is_lower_h or is_out_of_region
        if self.verbose or True:
            if is_lower_h:
                print('reached the h limit')
            if is_out_of_region:
                print('reached drawing area limit')
        return done

    def _do_action(self, action):
        rotation_i = action['rotation']
        length_i = action['length']
        direction_i = self.init_action['direction']
        grasp_width_i = action['grasp_width']
        self.med.gripper.move(grasp_width_i, 10.0)
        current_plane_pose = self.med.get_plane_pose()

        delta_pos_plane = length_i * np.array([np.cos(direction_i), np.sin(direction_i)])
        delta_h = -0.003
        delta_pos = np.append(delta_pos_plane, delta_h) # push down in the plane
        new_pos = current_plane_pose[:3] + delta_pos

        # new_pos = current_plane_pose[:3] + delta_pos
        # rotate along the x axis of the grasp frame
        rot_quat = tr.quaternion_about_axis(angle=rotation_i, axis=(1, 0, 0))
        new_quat = tr.quaternion_multiply(current_plane_pose[3:], rot_quat)
        new_pose = np.concatenate([new_pos, new_quat])
        reached = self.delta_cartesian_move(dx=delta_pos[0], dy=delta_pos[1], target_z=new_pos[2], quat=new_quat)

        # planning_result = self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', target_pose=list(new_pose), frame_id=self.med.drawing_frame, position_tol=0.0005, orientation_tol=0.001)

        # check if the planning has failed or if the execution has failed
        action_feedback = {
            # 'planning_success': planning_result.planning_result.success,
            # 'execution_success': planning_result.execution_result.success,
        }
        return action_feedback

    def delta_cartesian_move(self, dx, dy, target_z=None, quat=None, frame_id='grasp_frame'):
        arm_id = 0
        target_orientation = None
        cartesian_motion_frame_id = 'med_kuka_link_ee'
        if quat is not None:
            # Find the transformation to the med_kuka_link_ee --- Cartesian Impedance mode sets the pose with respect to the med_kuka_link_ee frame
            ee_pose_gf = self.med.get_current_pose(frame_id,
                                                   ref_frame=cartesian_motion_frame_id)  # IMPEDANCE ORIENTATION IN MED_KUKA_LINK_EE FRAME!
            # desired_quat = tr.quaternion_multiply(ee_pose_gf[3:], quat)
            desired_quat = tr.quaternion_multiply(quat, tr.quaternion_inverse(ee_pose_gf[3:]))
            target_orientation = Quaternion()
            target_orientation.x = desired_quat[0]
            target_orientation.y = desired_quat[1]
            target_orientation.z = desired_quat[2]
            target_orientation.w = desired_quat[3]

        if target_z is not None:
            wf_pose_gf = self.med.get_current_pose(frame_id, ref_frame=self.med.cartesian.sensor_frames[arm_id])
            wf_pose_ee = self.med.get_current_pose(cartesian_motion_frame_id, ref_frame=self.med.cartesian.sensor_frames[arm_id])
            delta_z = wf_pose_ee[2] - wf_pose_gf[2]
            target_z = target_z + delta_z
        reached = self.med.move_delta_cartesian_impedance(arm=arm_id, dx=dx, dy=dy, target_z=target_z, target_orientation=target_orientation)
        return reached