#! /usr/bin/env python
import os
import sys
import numpy as np
import threading
import copy
import rospy
import tf
import tf.transformations as tr
import mmint_utils

from arm_robots.med import Med
from mmint_camera_utils.point_cloud_parsers import PicoFlexxPointCloudParser
from mmint_camera_utils.tf_recording import save_tfs

from victor_hardware_interface.victor_utils import Stiffness
from victor_hardware_interface_msgs.msg import ControlMode, MotionStatus

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Vector3
from control_msgs.msg import FollowJointTrajectoryFeedback
from visualization_msgs.msg import Marker

package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_control')[0], 'bubble_control')

class BubbleCalibration(object):

    def __init__(self, object_topic='estimated_object', scene_name='calibration_scene', save_path=None):
        self.object_topic = object_topic
        self.scene_name = scene_name
        rospy.init_node('motion_test')
        self.lock = threading.Lock()
        self.save_path = self._get_save_path(save_path)
        self.tf_listener = tf.TransformListener()
        self.med = Med(display_goals=False)
        self.camera_name_right = 'pico_flexx_right'
        self.camera_name_left = 'pico_flexx_left'
        self.camera_parser_right = PicoFlexxPointCloudParser(camera_name=self.camera_name_right, scene_name=self.scene_name, save_path=self.save_path)
        self.camera_parser_left = PicoFlexxPointCloudParser(camera_name=self.camera_name_left, scene_name=self.scene_name, save_path=self.save_path)

        self.marker_pose = None
        self.path_to_tool_configuration = os.path.join(package_path, 'config', 'calibration_tool.yaml')
        self.cfg = mmint_utils.load_cfg(self.path_to_tool_configuration)
        self.calibration_home_conf = [-0.0009035506269389259, 0.36900237414385106, -0.5636396688935227, -1.3018021297139244, 0.19157857303422685, 1.5285128055113293, 1.035954690370297]
        self.grasp_forces = iter([10., 20, 30, 40])
        # Set up the robot
        self._setup()

    def _get_save_path(self, save_path=None):
        if save_path is None:
            # Get some default directory based on the current working directory
            save_path = os.path.join(package_path, 'calibration_data')
        else:
            if save_path.startswith("/"):
                save_path = save_path  # we provide the full path (absolute)
            else:
                exec_path = os.getcwd()
                save_path = os.path.join(exec_path, save_path)  # we save the data on the path specified (relative)
        return save_path

    def _setup(self):
        self.med.connect()
        self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
        self.med.set_grasping_force(8.0)
        # self.med.set_control_mode(ControlMode.JOINT_IMPEDANCE, stiffness=Stiffness.STIFF, vel=0.075)  # Low vel for safety
        # self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.05)  # Low vel for safety
        self._calibration_home()

    def _calibration_home(self):
        self.med.plan_to_joint_config(self.med.arm_group, self.calibration_home_conf)

    def _sample_calibration_pose(self):
        # TODO: Make this random
        h_gap = 0.00
        cal_position_i = [self.cfg['tool_center']['x'], self.cfg['tool_center']['y'], self.cfg['tool_size']['h']*.5+h_gap]
        cal_quat_i = tr.quaternion_from_euler(-np.pi, 0, np.pi)
        cal_quat_i = np.array([0, 1, 0, 0])
        cal_pose_i = np.concatenate([cal_position_i, cal_quat_i])
        cal_pose = cal_pose_i # todo: make this random
        return cal_pose

    def _sample_grasp_force(self):
        # grasp_force = np.random.uniform(10, 40)
        grasp_force = next(self.grasp_forces)
        return grasp_force

    def _plan_to_pose(self, pose, supervision=False):
        if supervision:
            self.med.set_execute(False)
        plan_result = self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', target_pose=list(pose), frame_id='med_base')
        if supervision:
            for i in range(10):
                k = input('Execute plan (y: yes, r: replan, f: finish')
                if k == 'y':
                    self.med.set_execute(True)
                    self.med.follow_arms_joint_trajectory(plan_result.planning_result.plan.joint_trajectory)
                    break
                elif k == 'r':
                    break
                elif k == 'f':
                    return
                else:
                    pass

    def calibrate(self, supervision=False):
        """
        Adjust the robot so the object has a constant pose (target_pose) in the reference ref_frame
        Args:
            target_pose: <list> pose as [x,y,z,qw,qx,qy,qz]
            ref_frame: <str>
        """
        self.med.open_gripper()
        calibration_pose = self._sample_calibration_pose()
        pre_pose_h_gap = 0.15
        pre_pose_delta = np.zeros_like(calibration_pose)
        pre_pose_delta[2] = pre_pose_h_gap
        pre_pose = calibration_pose + pre_pose_delta

        # Preposition the grasp
        self._plan_to_pose(pre_pose, supervision=supervision)
        # Position the grasp
        self._plan_to_pose(calibration_pose, supervision=supervision)
        # Grasp
        grasp_force = float(self._sample_grasp_force())
        print('Grasp force: ', grasp_force)
        self.med.set_grasping_force(grasp_force)
        self.med.grasp(self.cfg['tool_size']['diameter'], speed=10.)

        # TODO: Record grasp data
        self._record()
        # Move back to home position
        _ = input('press enter to move back to pose')
        self.med.open_gripper()
        # Preposition the grasp
        self._plan_to_pose(pre_pose, supervision=supervision)
        self._calibration_home()

    def _record(self):
        self.camera_parser_left.record()
        self.camera_parser_right.record()
        child_names = ['pico_flexx_left_link', 'pico_flexx_right_link', 'pico_flexx_left_optical_link', 'grasp_frame', 'calibration_tool']
        parent_names = 'med_base'
        tf_save_path = os.path.join(self.save_path, self.scene_name, 'tfs')
        save_tfs(child_names, parent_names, tf_save_path, file_name='recorded_tfs_{}'.format(self.camera_parser_left.counter['pointcloud']))

def calibration_test(supervision=False):
    bc = BubbleCalibration()
    for i in range(5):
        bc.calibrate(supervision=supervision)


if __name__ == '__main__':
    supervision = False
    calibration_test(supervision=supervision)