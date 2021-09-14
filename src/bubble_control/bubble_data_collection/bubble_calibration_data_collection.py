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
from bubble_control.bubble_data_collection.data_collector_base import DataCollectorBase

from bubble_control.bubble_data_collection.bubble_data_collection_base import BubbleDataCollectionBase

package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_control')[0], 'bubble_control')


class BubbleCalibrationDataCollection(BubbleDataCollectionBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.marker_pose = None
        self.path_to_tool_configuration = os.path.join(package_path, 'config', 'calibration_tool.yaml')
        self.cfg = mmint_utils.load_cfg(self.path_to_tool_configuration)
        self.calibration_home_conf = [-0.0009035506269389259, 0.36900237414385106, -0.5636396688935227,
                                      -1.3018021297139244, 0.19157857303422685, 1.5285128055113293, 1.035954690370297]
        # self.grasp_forces = [10., 20, 30, 40]
        self.grasp_forces = [10., 20]
        self._setup()

    def _get_med(self):
        return Med(display_goals=False)

    def _setup(self):
        self.med.connect()
        self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
        self.med.set_grasping_force(8.0)
        # self.med.set_control_mode(ControlMode.JOINT_IMPEDANCE, stiffness=Stiffness.STIFF, vel=0.075)  # Low vel for safety
        # self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.05)  # Low vel for safety
        self._calibration_home()

    def _calibration_home(self):
        self.med.plan_to_joint_config(self.med.arm_group, self.calibration_home_conf)

    def _get_legend_column_names(self):
        column_names = ['UndeformedFC', 'DeformedFC',  'GraspForce', 'GraspWidth', 'CalibrationToolSize', 'CalibrationToolPose']
        return column_names

    def _sample_calibration_pose(self):
        # TODO: Make this random
        h_gap = 0.00
        cal_position_i = [self.cfg['tool_center']['x'], self.cfg['tool_center']['y'], self.cfg['tool_size']['h']*.5+h_gap]
        cal_quat_i = tr.quaternion_from_euler(-np.pi, 0, np.pi)
        cal_quat_i_base = np.array([0, 1, 0, 0])
        delta_angle = np.random.uniform(-90, 90)
        delta_orientation_quat = tr.quaternion_about_axis(angle=np.deg2rad(delta_angle), axis=[1,0,0])
        cal_quat_i = tr.quaternion_multiply(delta_orientation_quat, cal_quat_i_base)
        cal_pose_i = np.concatenate([cal_position_i, cal_quat_i])
        cal_pose = cal_pose_i # todo: make this random
        return cal_pose

    def _sample_grasp_forces(self):
        # grasp_forces = np.random.uniform(10, 40, 4, dtype=np.float)
        grasp_forces = self.grasp_forces
        grasp_forces = [float(gf) for gf in grasp_forces]
        return grasp_forces

    def _get_legend_lines(self, data_params):
        legend_lines = []
        for i, undef_fc_i in enumerate(data_params['undeformed_fc']):
            def_fc_i = data_params['deformed_fc'][i]
            grasp_force_i = data_params['grasp_force'][i]
            grasp_width_i = data_params['grasp_width'][i]
            line_i = [undef_fc_i, def_fc_i,  grasp_force_i, grasp_width_i, data_params['calibration_tool_size'], data_params['calibration_tool_pose']]
            legend_lines.append(line_i)
        return legend_lines

    def _collect_data_sample(self, params=None):
        """
        Adjust the robot so the object has a constant pose (target_pose) in the reference ref_frame
        returns:
            data_params: <dict> containing the parameters and information of the collected data
        """
        data_params_keys = ['undeformed_fc', 'deformed_fc', 'grasp_force', 'grasp_width', 'calibration_tool_pose', 'calibration_tool_size']
        data_params = dict(zip(data_params_keys, [[] for x in data_params_keys]))
        self.med.open_gripper()
        calibration_pose = self._sample_calibration_pose()
        pre_pose_h_gap = 0.15
        pre_pose_delta = np.zeros_like(calibration_pose)
        pre_pose_delta[2] = pre_pose_h_gap
        pre_pose = calibration_pose + pre_pose_delta

        data_params['calibration_tool_size'] = self.cfg['tool_size']
        data_params['calibration_tool_pose'] = self.cfg['tool_center']

        # Preposition the grasp
        # self._plan_to_pose(pre_pose, supervision=self.supervision)
        # Position the grasp
        self._plan_to_pose(calibration_pose, supervision=self.supervision)

        grasp_forces = self._sample_grasp_forces()
        # if we return multiple forces, collect them for the same position
        for i, grasp_force_i in enumerate(grasp_forces):
            # Record bubble state without deformation
            undef_fc = (self.filecode+i)*2-1
            self._record(fc=undef_fc)
            data_params['undeformed_fc'].append(undef_fc)
            # Grasp
            print('Grasp force: ', grasp_force_i)
            self.med.set_grasping_force(grasp_force_i)
            grasp_width = self.cfg['tool_size']['diameter']

            data_params['grasp_force'].append(grasp_force_i)
            data_params['grasp_width'].append(grasp_width)

            self.med.grasp(grasp_width, speed=10.)
            # rospy.sleep(2.0)
            def_fc = (self.filecode+i)*2
            self._record(fc=def_fc)
            data_params['deformed_fc'].append(def_fc)
            # Move back to home position
            # _ = input('press enter to move back to pose')
            self.med.open_gripper()
        # Preposition the grasp back
        self._plan_to_pose(pre_pose, supervision=self.supervision)
        self._calibration_home()

        return data_params