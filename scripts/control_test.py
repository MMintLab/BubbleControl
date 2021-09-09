#! /usr/bin/env python
import os
import sys
import numpy as np
import threading
import copy
import rospy
import tf
import tf.transformations as tr

from arm_robots.med import Med

from victor_hardware_interface.victor_utils import Stiffness
from victor_hardware_interface_msgs.msg import ControlMode, MotionStatus

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Vector3
from control_msgs.msg import FollowJointTrajectoryFeedback
from visualization_msgs.msg import Marker
from mmint_utils.terminal_colors import term_colors


class BubbleController(object):

    def __init__(self, object_topic='estimated_object'):
        self.object_topic = object_topic
        rospy.init_node('motion_test')
        self.lock = threading.Lock()
        self.tf_listener = tf.TransformListener()
        self.med = Med(display_goals=False)
        self.marker_pose = None
        self.pose_subscriber = rospy.Subscriber(self.object_topic, Marker, self.marker_callback)
        self.tf_broadcaster = tf.TransformBroadcaster()
        # Set up the robot
        self._setup()

    def _setup(self):
        self.med.connect()
        self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)
        # self.med.set_control_mode(ControlMode.JOINT_IMPEDANCE, stiffness=Stiffness.STIFF, vel=0.075)  # Low vel for safety
        # self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.05)  # Low vel for safety

    def marker_callback(self, data):
        self.lock.acquire()
        try:
            pose = [data.pose.position.x,
                    data.pose.position.y,
                    data.pose.position.z,
                    data.pose.orientation.x,
                    data.pose.orientation.y,
                    data.pose.orientation.z,
                    data.pose.orientation.w
                    ]
            self.marker_pose = {
                'pose': pose,
                'frame': data.header.frame_id,
            }
        finally:
            self.lock.release()

    def control(self, desired_pose, ref_frame, supervision=False):
        """
        Adjust the robot so the object has a constant pose (target_pose) in the reference ref_frame
        Args:
            target_pose: <list> pose as [x,y,z,qw,qx,qy,qz]
            ref_frame: <str>
        """

        T_desired = tr.quaternion_matrix(desired_pose[3:])
        T_desired[:3, 3] = desired_pose[:3]
        try:
            while self.marker_pose is None:
                rospy.sleep(0.1)

            print(
            f'''
        *************************************************
             
             ---- Control: {term_colors.GREEN} ON {term_colors.ENDC} ----
             
             Stabilizing the tool pose
             
        *************************************************
            '''
            )
            while not rospy.is_shutdown():

                # Read object position
                self.lock.acquire()
                try:
                    current_marker_pose = copy.deepcopy(self.marker_pose)
                finally:
                    self.lock.release()
                T_mf = tr.quaternion_matrix(current_marker_pose['pose'][3:])
                T_mf[:3,3] = current_marker_pose['pose'][:3]
                T_mf_desired = T_desired @ np.linalg.inv(T_mf)   # maybe it is this

                # Compute the target
                target_pose = np.concatenate([T_mf_desired[:3,3], tr.quaternion_from_matrix(T_mf_desired)])
                # broadcast target_pose:
                self.tf_broadcaster.sendTransform(list(target_pose[:3]), list(target_pose[3:]), rospy.Time.now(), '{}_desired'.format(current_marker_pose['frame']), ref_frame)
                self.tf_broadcaster.sendTransform(list(desired_pose[:3]), list(desired_pose[3:]), rospy.Time.now(), 'desired_obj_pose', ref_frame)
                self.tf_broadcaster.sendTransform(list(current_marker_pose['pose'][:3]), list(current_marker_pose['pose'][3:]), rospy.Time.now(), 'current_obj_pose', current_marker_pose['frame'])
                if supervision:
                    self.med.set_execute(False)
                plan_result = self.med.plan_to_pose(self.med.arm_group, current_marker_pose['frame'], target_pose=list(target_pose), frame_id=ref_frame)
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
        except rospy.ROSInterruptException:
            pass

def control_test(supervision=False):
    bc = BubbleController()
    target_pose = np.array([0.5, 0, .5, 0, -0.7071, 0.7071, 0])
    bc.control(target_pose, ref_frame='med_base', supervision=supervision)


if __name__ == '__main__':
    supervision = False
    control_test(supervision=supervision)