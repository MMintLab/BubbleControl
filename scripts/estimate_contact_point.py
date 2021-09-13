#! /usr/bin/env python
import os
import pdb
import sys
import numpy as np
import threading
import copy
import rospy
import tf
import tf.transformations as tr

from arm_robots.med import Med
from arc_utilities.listener import Listener
import tf2_geometry_msgs  # Needed by TF2Wrapper
from arc_utilities.tf2wrapper import TF2Wrapper
from victor_hardware_interface.victor_utils import Stiffness
from victor_hardware_interface_msgs.msg import ControlMode, MotionStatus

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Vector3, WrenchStamped
from control_msgs.msg import FollowJointTrajectoryFeedback
from visualization_msgs.msg import Marker


class ContactPointEstimator(object):

    def __init__(self, object_topic='estimated_object', plane_frame='med_base', rate=10):
        self.object_topic = object_topic
        self.plane_frame = plane_frame
        self.rate = rate
        try:
            rospy.init_node('contact_point_estimator')
        except (rospy.exceptions.ROSInitException, rospy.exceptions.ROSException):
            pass
        self.tf2_listener = TF2Wrapper()
        self.pose_listener = Listener(self.object_topic, Marker, wait_for_data=False)
        self.estimate_contact_point()
        # rospy.spin()

    def get_marker_pose(self):
        data = self.pose_listener.get(block_until_data=True)
        pose = [data.pose.position.x,
                data.pose.position.y,
                data.pose.position.z,
                data.pose.orientation.x,
                data.pose.orientation.y,
                data.pose.orientation.z,
                data.pose.orientation.w
                ]
        marker_pose = {
            'pose': pose,
            'frame': data.header.frame_id,
        }
        return marker_pose

    def _estimate_contact_point(self):
        marker_pose = self.get_marker_pose()
        marker_translation = marker_pose['pose'][:3]
        marker_quat = marker_pose['pose'][3:]
        marker_axis = np.array([0,0,1])
        plane_normal_axis = np.array([0,0,1])  # axis normal to the plane in plane_frame

        marker_axis_mpf = tr.quaternion_matrix(marker_quat)[:3,:3] @ marker_axis# in the marker_pose frame
        self.tf2_listener.send_transform(translation=marker_translation, quaternion=marker_quat, parent=marker_pose['frame'], child='tool_frame', is_static=False)
        tool_pose_pf = self.tf2_listener.get_transform(parent=self.plane_frame, child='tool_frame')
        marker_parent_frame_pf = self.tf2_listener.get_transform(parent=self.plane_frame, child=marker_pose['frame'])

        contact_point_pf = self._get_contact_point_plane_frame(tool_pose_pf, marker_axis, plane_normal_axis)
        parent_frame_pf = self._get_contact_point_plane_frame(marker_parent_frame_pf, np.array([0,0,1]) , plane_normal_axis)
        self.tf2_listener.send_transform(translation=contact_point_pf, quaternion=[0,0,0,1], parent=self.plane_frame, child='tool_contact_point', is_static=False)
        self.tf2_listener.send_transform(translation=parent_frame_pf, quaternion=[0,0,0,1], parent=self.plane_frame, child='tool_contact_point_fake', is_static=False)

    def _get_contact_point_plane_frame(self, pose_pf, tool_axis, plane_normal_axis):
        h = pose_pf[2, 3]
        marker_axis_pf = pose_pf[:3, :3] @ tool_axis  # in the plane frame
        cos_angle = np.dot(marker_axis_pf, plane_normal_axis)
        angle = np.rad2deg(np.arccos(cos_angle))  # todo: remove (debugging)
        dist = -h / cos_angle
        contact_point_pf = pose_pf[:3, 3] + dist * marker_axis_pf
        return contact_point_pf

    def estimate_contact_point(self):
        while not rospy.is_shutdown():
            try:
                self._estimate_contact_point()
                rospy.Rate(self.rate)
            except (rospy.ROSInterruptException, rospy.ROSException) as e:
                print(e)
                return None



if __name__ == '__main__':
    cpe = ContactPointEstimator()
