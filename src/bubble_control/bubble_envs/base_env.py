from gym import Env
import rospy
import tf2_ros as tf2
from abc import abstractmethod

from arc_utilities.listener import Listener
from arc_utilities.tf2wrapper import TF2Wrapper

from mmint_camera_utils.tf_utils.tf_utils import get_tfs
from mmint_camera_utils.recording_utils.data_recording_wrappers import TFSelfSavedWrapper
from mmint_camera_utils.recording_utils.data_recording_wrappers import DictSelfSavedWrapper
from geometry_msgs.msg import WrenchStamped
from bubble_utils.bubble_data_collection.wrench_recorder import WrenchRecorder
from bubble_utils.bubble_med.bubble_med import BubbleMed
from bubble_utils.bubble_parsers.bubble_parser import BubbleParser


class BaseEnv(Env):
    """
    Main Attributes:
     - action_space:
     - observation_space:
     - reward_range: (by default is (-inf, inf))
    Main methods:
     - step(action):
     - reset():
     - render():
     - close():
    For more information about the Env class, check: https://github.com/openai/gym/blob/master/gym/core.py
    """
    def __init__(self):
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.num_steps = 0

    @classmethod
    def get_name(cls):
        return 'base_env'

    @abstractmethod
    def _get_action_space(self):
        pass

    @abstractmethod
    def _get_observation_space(self):
        pass

    @abstractmethod
    def _get_observation(self):
        pass

    def _do_action(self, a):
        return {}

    def is_action_valid(self, a):
        # by default, all actions are valid.
        # Override this method if some actions are not valid
        return True

    def _is_done(self, observation, a):
        return False

    def get_observation(self):
        obs = self._get_observation()
        return obs

    def get_action(self):
        valid_action = False
        action = None
        for i in range(1000):
            action = self.action_space.sample()
            valid_action = self.is_action_valid(action)
            if valid_action:
                break
        return action, valid_action

    def step(self, a):
        # This is just the basic layout. It can be extended on subclasses
        info = {}
        action_feedback = self._do_action(a)
        info.update(action_feedback)
        observation = self.get_observation()
        done = self._is_done(observation, a)
        reward = self._get_reward(a, observation)
        self.num_steps += 1
        return observation, reward, done, info

    def render(self):
        pass

    def _get_reward(self, a, observation):
        return 0

    def reset(self):
        self.num_steps = 0


class MedBaseEnv(BaseEnv):
    """
    Environment with:
     - Med robot
     - Wrench listener
     - Tf listener
    """
    def __init__(self, wrench_topic='/med/wrench', save_path=None, scene_name='default_scene', wrap_data=False, verbose=False, buffered=False):
        self.wrench_topic = wrench_topic
        self.save_path = self._get_save_path(save_path)
        self.scene_name = scene_name
        self.wrap_data = wrap_data
        self.verbose = verbose
        self.buffered = buffered
        self._init_ros_node()
        self.tf_buffer = tf2.Buffer()
        self.tf_listener = tf2.TransformListener(buffer=self.tf_buffer, queue_size=1000, buff_size=500000)
        self.tf2_listener = TF2Wrapper(buffer=self.tf_buffer, listener=self.tf_listener)
        self.wrench_listener = Listener(self.wrench_topic, WrenchStamped, wait_for_data=True)
        self.med = self._get_med()
        self.wrench_recorder = WrenchRecorder(self.wrench_topic, scene_name=self.scene_name, save_path=self.save_path, wrap_data=self.wrap_data)
        super().__init__()

    @classmethod
    def get_name(cls):
        return 'med_base_env'

    def _get_save_path(self, save_path):
        if save_path is None:
            save_path = '/tmp/{}_data'.format(self.get_name())
        return save_path

    def _get_med(self):
        med = BubbleMed(display_goals=False)
        med.connect()
        return med

    def _init_ros_node(self):
        try:
            rospy.init_node('{}_node'.format(self.get_name()))
        except (rospy.exceptions.ROSInitException, rospy.exceptions.ROSException):
            pass

    def _plan_to_pose(self, pose, frame_id='med_base', supervision=False, stop_condition = None):
        plan_success = False
        execution_success = False
        plan_found = False
        while (not rospy.is_shutdown()) and not plan_found:
            if supervision:
                self.med.set_execute(False)
            plan_result = self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', target_pose=list(pose),
                                                frame_id=frame_id, stop_condition=stop_condition)
            plan_success = plan_result.success
            execution_success = plan_result.execution_result.success
            if not plan_success:
                print('@' * 20 + '    Plan Failed    ' + '@' * 20)
                import pdb;
                pdb.set_trace()
            if supervision or not plan_success:
                for i in range(10):
                    k = input('Execute plan (y: yes, r: replan, f: finish): ')
                    if k == 'y':
                        self.med.set_execute(True)
                        execution_result = self.med.follow_arms_joint_trajectory(
                            plan_result.planning_result.plan.joint_trajectory)
                        execution_success = execution_result.success
                        plan_found = True
                        break
                    elif k == 'r':
                        break
                    elif k == 'f':
                        return
                    else:
                        pass
            else:
                plan_found = True

        if not execution_success:
            # It seems tha execution always fails (??)
            print('-' * 20 + '    Execution Failed    ' + '-' * 20)

        return plan_success, execution_success

    def _get_tf_frames(self):
        tf_frames = ['grasp_frame', 'med_kuka_link_ee', 'wsg50_finger_left', 'wsg50_finger_right']
        return tf_frames

    def _get_wrench_frames(self):
        wrench_frames = ['grasp_frame', 'med_base']
        return wrench_frames

    def _get_wrench(self):
        frame_names = self._get_wrench_frames()
        wrench = self.wrench_recorder.get_wrench(frame_names=frame_names)
        return wrench

    def _get_tfs(self):
        tf_frames = self._get_tf_frames()
        parent_names = 'med_base'
        tfs = get_tfs(tf_frames, parent_names, verbose=self.verbose, buffer=self.tf_buffer) # df of the frames
        if self.wrap_data:
            tfs = TFSelfSavedWrapper(tfs, data_params={'save_path': self.save_path, 'scene_name': self.scene_name})
        return tfs

    def get_observation(self):
        obs = self._get_observation()
        if self.wrap_data:
            obs = DictSelfSavedWrapper(obs, data_params={'save_path': self.save_path, 'scene_name': self.scene_name})
        return obs


class BubbleBaseEnv(MedBaseEnv):
    """
    Environment with:
     - Med robot
     - Wrench listener
     - Tf listener
     - PicoFlexxParsers for each bubble
    """
    def __init__(self, *args , right=True, left=True, record_shear=False, **kwargs):
        self.right = right
        self.left = left
        self.record_shear = record_shear
        super().__init__(*args, **kwargs)
        if self.right:
            self.camera_name_right = 'pico_flexx_right'
            self.camera_parser_right = BubbleParser(camera_name=self.camera_name_right,
                                                                 scene_name=self.scene_name, save_path=self.save_path,
                                                                 verbose=False, wrap_data=self.wrap_data, record_shear=record_shear, buffered=self.buffered)
        if self.left:
            self.camera_name_left = 'pico_flexx_left'
            self.camera_parser_left = BubbleParser(camera_name=self.camera_name_left,
                                                                scene_name=self.scene_name, save_path=self.save_path,
                                                                verbose=False, wrap_data=self.wrap_data, record_shear=record_shear, buffered=self.buffered)

    @classmethod
    def get_name(cls):
        return 'bubble_base_env'

    def _get_tf_frames(self):
        frames = super()._get_tf_frames()
        if self.left:
            frames += ['pico_flexx_left_link', 'pico_flexx_left_optical_frame']
        if self.right:
            frames += ['pico_flexx_right_link', 'pico_flexx_right_optical_frame']
        return frames

    def _get_bubble_observation(self):
        bubble_observation = {}
        if self.right:
            obs_r = self._get_observation_from_camera_parser(self.camera_parser_right)
            for k, v in obs_r.items():
                bubble_observation['bubble_{}_right'.format(k)] = v
        if self.left:
            obs_l = self._get_observation_from_camera_parser(self.camera_parser_left)
            for k, v in obs_l.items():
                bubble_observation['bubble_{}_left'.format(k)] = v
        if self.wrap_data:
            bubble_observation = DictSelfSavedWrapper(bubble_observation)
        return bubble_observation

    def _get_observation_from_camera_parser(self, camera_parser):
        obs = {}
        obs['camera_info_color'] = camera_parser.get_camera_info_color()
        obs['camera_info_depth'] = camera_parser.get_camera_info_depth()
        obs['color_img'] = camera_parser.get_image_color()
        obs['depth_img'] = camera_parser.get_image_depth()
        obs['point_cloud'] = camera_parser.get_point_cloud()
        if self.record_shear and isinstance(camera_parser, BubbleParser):
            obs['shear_deformation'] = camera_parser.get_shear_deformation()
            obs['shear_img'] = camera_parser.get_shear_image()
        return obs
