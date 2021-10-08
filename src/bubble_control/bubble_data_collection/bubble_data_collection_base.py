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
from collections import defaultdict

from arm_robots.med import Med
from arc_utilities.listener import Listener
import tf2_geometry_msgs  # Needed by TF2Wrapper
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
        self.counter = 0

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
        if fc is None:
            fc = self.counter + 1
        wrench = self.wrench_listener.get(block_until_data=True)
        wrenches = [wrench]
        if frame_names is not None:
        # record only the frame that the topic is published to
            for frame_name_i in frame_names:
                # import pdb; pdb.set_trace()
                wrench_frame_i = self.tf2_listener.transform_to_frame(wrench, target_frame=frame_name_i, timeout=rospy.Duration(secs=1))
                wrenches.append(wrench_frame_i)

        df = self._pack_wrenches(wrenches)
        # save them on a dataframe
        filename = '{}_wrench_{:06d}'.format(self.scene_name, fc)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        full_save_path = os.path.join(self.save_path, '{}.csv'.format(filename))
        df.to_csv(full_save_path, index=False)
        self.counter += 1

    def _pack_wrenches(self, wrenches):
        wrenches_dict = defaultdict(list)
        for i, wrench_i in enumerate(wrenches):
            wrench_dict_i = self._pack_message_as_dict(wrench_i)
            for k, v in wrench_dict_i.items():
                wrenches_dict[k].append(v)
        wrenches_df = pd.DataFrame(wrenches_dict)
        return wrenches_df

    def _pack_message_as_dict(self, msg):
        output_dict = {}
        for slot_name in msg.__slots__:
            slot_value = getattr(msg, slot_name)
            if isinstance(slot_value, rospy.rostime.Time):
                keys = ['secs', 'nsecs']
                sub_dict_updated = {}
                for k in keys:
                    sub_dict_updated['{}.{}'.format(slot_name, k)] = getattr(slot_value, k)
                output_dict.update(sub_dict_updated)
            elif '__slots__' in slot_value.__dir__():
                sub_dict = self._pack_message_as_dict(slot_value)
                # update keys:
                sub_dict_updated = {}
                for k, v in sub_dict.items():
                    sub_dict_updated['{}.{}'.format(slot_name, k)] = v
                output_dict.update(sub_dict_updated)
            else:
                output_dict[slot_name] = slot_value

        return output_dict


class MedDataCollectionBase(DataCollectorBase):

    def __init__(self, data_path=None, supervision=False, scene_name='med_data_collection', wrench_topic='/med/wrench'):
        super().__init__(data_path=data_path)
        self.supervision = supervision
        self.scene_name = scene_name
        self.wrench_topic = wrench_topic
        self.lock = threading.Lock()
        self.save_path = self.data_path # rename for backward compatibility
        # Init ROS node
        try:
            rospy.init_node(self.scene_name)
        except (rospy.exceptions.ROSInitException, rospy.exceptions.ROSException):
            pass
        self.tf_listener = tf.TransformListener()
        self.tf2_listener = TF2Wrapper()
        self.wrench_listener = Listener(self.wrench_topic, WrenchStamped, wait_for_data=True)
        self.med = self._get_med()

        self.wrench_recorder = WrenchRecorder(self.wrench_topic, scene_name=self.scene_name, save_path=self.save_path)

    def _get_med(self):
        med = Med(display_goals=False)
        med.connect()
        return med

    def _plan_to_pose(self, pose, supervision=False):
        plan_success = False
        execution_success = False
        plan_found = False
        while (not rospy.is_shutdown()) and not plan_found:
            if supervision:
                self.med.set_execute(False)
            plan_result = self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', target_pose=list(pose),
                                                frame_id='med_base')
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
            # It seams tha execution always fails (??)
            print('-' * 20 + '    Execution Failed    ' + '-' * 20)

        return plan_success, execution_success

    def _get_recording_frames(self):
        child_frames = ['grasp_frame', 'med_kuka_link_ee']
        return child_frames

    def _record(self, fc=None):
        self.wrench_recorder.record(fc=fc, frame_names=['grasp_frame', 'med_base'])
        child_names = self._get_recording_frames()
        parent_names = 'med_base'
        tf_save_path = os.path.join(self.save_path, self.scene_name, 'tfs')
        if fc is None:
            tf_fc = self.wrench_recorder.counter
        else:
            tf_fc = fc
        save_tfs(child_names, parent_names, tf_save_path, file_name='recorded_tfs_{:06d}'.format(tf_fc))


class BubbleDataCollectionBase(MedDataCollectionBase):

    def __init__(self, data_path=None, supervision=False, scene_name='bubble_data_collection', wrench_topic='/med/wrench'):
        super().__init__(data_path=data_path, supervision=supervision, scene_name=scene_name, wrench_topic=wrench_topic)
        self.camera_name_right = 'pico_flexx_right'
        self.camera_name_left = 'pico_flexx_left'
        self.camera_parser_right = PicoFlexxPointCloudParser(camera_name=self.camera_name_right,
                                                             scene_name=self.scene_name, save_path=self.save_path)
        self.camera_parser_left = PicoFlexxPointCloudParser(camera_name=self.camera_name_left,
                                                            scene_name=self.scene_name, save_path=self.save_path)


    def _get_recording_frames(self):
        super_frames = super()._get_recording_frames()
        child_frames = super_frames + ['pico_flexx_left_link', 'pico_flexx_right_link', 'pico_flexx_left_optical_frame', 'pico_flexx_right_optical_frame']
        return child_frames

    def _record(self, fc=None):
        super()._record(fc=fc)
        self.camera_parser_left.record(fc=fc)
        self.camera_parser_right.record(fc=fc)

