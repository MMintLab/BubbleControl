import abc
import os
import sys
import numpy as np
import threading
import copy
import rospy
import tf
import pandas as pd
import tf.transformations as tr
import mmint_utils

from arm_robots.med import Med
from arc_utilities.listener import Listener
from arc_utilities.tf2wrapper import TF2Wrapper

from mmint_camera_utils.point_cloud_parsers import PicoFlexxPointCloudParser
from mmint_camera_utils.tf_recording import save_tfs

from victor_hardware_interface.victor_utils import Stiffness
from victor_hardware_interface_msgs.msg import ControlMode, MotionStatus

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Vector3, WrenchStamped
from control_msgs.msg import FollowJointTrajectoryFeedback
from visualization_msgs.msg import Marker
from bubble_control.bubble_data_collection.data_collector_base import DataCollectorBase


package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_control')[0], 'bubble_control')

class WrenchRecorder(object):
    def __init__(self, wrench_topic, scene_name='scene', save_path=None):
        self.wrench_topic = wrench_topic
        self.scene_name = scene_name
        self.save_path = self._get_save_path(save_path)
        self.wrench_listener = Listener(self.wrench_topic, WrenchStamped, wait_for_data=True)
        self.tf2_listener = TF2Wrapper()

    def _get_save_path(self, save_path=None):
        if save_path is None:
            # Get some default directory based on the current working directory
            save_path = os.path.join(package_path, 'wrench_data')
        else:
            if save_path.startswith("/"):
                save_path = save_path  # we provide the full path (absolute)
            else:
                exec_path = os.getcwd()
                save_path = os.path.join(exec_path, save_path)  # we save the data on the path specified (relative)
        extended_save_path = os.path.join(save_path, self.scene_name, 'wrenches')
        return extended_save_path

    def record(self, fc=None, frame_names=None):

        wrench = self.wrench_listener.get(block_until_data=True)
        import pdb; pdb.set_trace()
        wrenches = []
        if frame_names is not None:
        # record only the frame that the topic is published to
            for frame_name_i in frame_names:
                wrench_frame_i = self.tf2_listener.transform_to_frame(wrench, target_frame=frame_name_i)
                wrenches.append(wrench_frame_i)

        df = self._pack_wrenches(wrenches)
        # save them on a dataframe
        filename = '{}_wrench_{:06d}'.format(self.scene_name, fc)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        full_save_path = os.path.join(self.save_path, '{}.csv'.format(filename))
        df.to_csv(full_save_path, index=False)

    def _pack_wrenches(self, wrenches):
        wrenches_data = []
        columns = ['frame'] + ['force_x', 'force_y', 'force_z', 'tau_x', 'tau_y', 'tau_z']
        import pdb; pdb.set_trace()
        for i, wrench_i in enumerate(wrenches):
            frame_i = None
        wrenches_df = pd.DataFrame(wrenches_data, columns=columns)
        return wrenches_df


class BubbleDataCollectionBase(DataCollectorBase):

    def __init__(self, data_path=None, supervision=False, scene_name='bubble_data_collection', wrench_topic='/med/wrench'):
        super().__init__(data_path=data_path)
        self.supervision = supervision
        self.scene_name = scene_name
        self.wrench_topic = wrench_topic
        self.lock = threading.Lock()
        self.save_path = self.data_path # rename for backward compatibility
        rospy.init_node(self.scene_name)
        self.tf_listener = tf.TransformListener()
        self.tf2_listener = TF2Wrapper()
        self.wrench_listener = Listener(self.wrench_topic, WrenchStamped, wait_for_data=True)
        self.med = self._get_med()
        self.camera_name_right = 'pico_flexx_right'
        self.camera_name_left = 'pico_flexx_left'
        self.camera_parser_right = PicoFlexxPointCloudParser(camera_name=self.camera_name_right,
                                                             scene_name=self.scene_name, save_path=self.save_path)
        self.camera_parser_left = PicoFlexxPointCloudParser(camera_name=self.camera_name_left,
                                                            scene_name=self.scene_name, save_path=self.save_path)

        self.wrench_recorder = WrenchRecorder(self.wrench_topic, scene_name=self.scene_name, save_path=self.save_path)

    @abc.abstractmethod
    def _get_med(self):
        pass

    def _plan_to_pose(self, pose, supervision=False):
        if supervision:
            self.med.set_execute(False)
        plan_result = self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', target_pose=list(pose), frame_id='med_base')
        if supervision:
            for i in range(10):
                k = input('Execute plan (y: yes, r: replan, f: finish): ')
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

    def _record(self, fc=None):
        # TODO: Record forces
        self.wrench_recorder.record(fc=fc, frame_names=['grasp_frame', 'med_base'])
        self.camera_parser_left.record(fc=fc)
        self.camera_parser_right.record(fc=fc)
        child_names = ['pico_flexx_left_link', 'pico_flexx_right_link', 'pico_flexx_left_optical_frame', 'pico_flexx_right_optical_frame', 'grasp_frame', 'med_kuka_link_ee', 'calibration_tool']
        parent_names = 'med_base'
        tf_save_path = os.path.join(self.save_path, self.scene_name, 'tfs')
        if fc is None:
            tf_fc = self.camera_parser_left.counter['pointcloud']-1
        else:
            tf_fc = fc
        save_tfs(child_names, parent_names, tf_save_path, file_name='recorded_tfs_{:06d}'.format(tf_fc))



