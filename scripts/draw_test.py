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
from arc_utilities.listener import Listener
import tf2_geometry_msgs  # Needed by TF2Wrapper
from arc_utilities.tf2wrapper import TF2Wrapper
from victor_hardware_interface.victor_utils import Stiffness
from victor_hardware_interface_msgs.msg import ControlMode, MotionStatus

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Vector3, WrenchStamped
from control_msgs.msg import FollowJointTrajectoryFeedback
from visualization_msgs.msg import Marker


class BubbleDrawer(object):

    def __init__(self, object_topic='estimated_object', wrench_topic='/med/wrench', force_threshold=5.):
        self.object_topic = object_topic
        self.wrench_topic = wrench_topic
        self.force_threshold = force_threshold
        rospy.init_node('drawing_test')
        self.tf_listener = tf.TransformListener()
        self.med = Med(display_goals=False)
        self.marker_pose = None
        self.calibrated_wrench = None
        self.pose_listener = Listener(self.object_topic, Marker, wait_for_data=False)
        self.wrench_listener = Listener(self.wrench_topic, WrenchStamped, wait_for_data=True)
        self.tf2_listener = TF2Wrapper()
        self.tf_broadcaster = tf.TransformBroadcaster()
        # Set up the robot
        self._setup()

    def home_robot(self):
        self.med.plan_to_joint_config(self.med.arm_group, [0, 1.0, 0, -0.8, 0, 0.9, 0])

    def _setup(self):
        self.med.connect()
        self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)

        # Set the robot to a favorable position so we can start the impedance mode
        self.home_robot()
        _ =  input('ok')
        # self.med.plan_to_pose(self.med.arm_group, self.med.wrist, target_pose=[0.6, 0.0, 0.4, 0.0, np.pi - 0.2, 0.0], frame_id="world")
        # self.med.set_control_mode(ControlMode.JOINT_IMPEDANCE, stiffness=Stiffness.STIFF, vel=0.075)  # Low vel for safety
        # self.med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.05)  # Low vel for safety

        self.calibrated_wrench = self._get_wrench()

    def _get_wrench(self):
        wrench_stamped_wrist = self.wrench_listener.get(block_until_data=True)
        wrench_stamped = self.tf2_listener.transform_to_frame(wrench_stamped_wrist, target_frame="world",
                                                              timeout=rospy.Duration(nsecs=int(5e8)))
        return wrench_stamped

    def _stop_signal(self, feedback):
        wrench_stamped = self._get_wrench()
        measured_fz = wrench_stamped.wrench.force.z
        calibrated_fz = measured_fz-self.calibrated_wrench.wrench.force.z
        flag_force = np.abs(calibrated_fz) >= np.abs(self.force_threshold)
        print('force z: {} (measured: {}) --- flag: {} ({} | {})'.format(calibrated_fz, measured_fz, flag_force, np.abs(calibrated_fz), np.abs(self.force_threshold)))
        return flag_force

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


    def draw_points(self, xy_points, ):
        draw_quat = np.array([-np.cos(np.pi/4), np.cos(np.pi/4), 0, 0])

        # first plan to the first corner
        pre_height = 0.2
        draw_height = 0.08  # set 0.1 for real
        pre_position = np.insert(xy_points[0], 2, pre_height)
        pre_pose = np.concatenate([pre_position, draw_quat], axis=0)
        self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', target_pose=list(pre_pose), frame_id='med_base')
        _ = input('press enter to continue')
        # self.med.set_control_mode(ControlMode.JOINT_IMPEDANCE, stiffness=Stiffness.STIFF, vel=0.075)  # Low vel for safety
        self.med.set_control_mode(ControlMode.JOINT_IMPEDANCE, stiffness=Stiffness.STIFF,
                                  vel=0.03)  # Low vel for safety
        # lower down:
        position_0 = np.insert(xy_points[0], 2, draw_height)
        pose_0 = np.concatenate([position_0, draw_quat], axis=0)
        self.med.set_execute(False)
        plan_result = self.med.plan_to_position_cartesian(self.med.arm_group, 'grasp_frame',
                                                          target_position=list(position_0))
        self.med.set_execute(True)
        self.med.follow_arms_joint_trajectory(plan_result.planning_result.plan.joint_trajectory,
                                              stop_condition=self._stop_signal)
        # read force
        first_contact_wrench = self._get_wrench()
        print('contact wrench: ', first_contact_wrench.wrench)
        self.force_threshold = 10
        self.med.set_control_mode(ControlMode.JOINT_IMPEDANCE, stiffness=Stiffness.STIFF, vel=0.1)
        for i, corner_i in enumerate(xy_points):
            position_i = np.insert(corner_i, 2, draw_height)
            pose_i = np.concatenate([pre_position, draw_quat], axis=0)
            # self.med.set_execute(False)
            print('Position: ', position_i)
            plan_result = self.med.plan_to_position_cartesian(self.med.arm_group, 'grasp_frame',
                                                              target_position=list(position_i))
            # self.med.set_execute(True)
            # self.med.follow_arms_joint_trajectory(plan_result.planning_result.plan.joint_trajectory, stop_condition=self._stop_signal)
            _ = input('press enter to continue')
        self.med.plan_to_pose(self.med.arm_group, 'grasp_frame', target_pose=list(pre_pose), frame_id='med_base')
        self.home_robot()

    def draw_square(self, side_size=0.2, center=(0.55, -0.1)):
        corners = np.asarray(center) + side_size * 0.5 * np.array([[1, 1],[1, -1], [-1, -1], [-1, 1], [1,1]])
        self.draw_points(corners)

    def draw_regular_polygon(self, num_sides, inscribed_radius=0.2, center=(0.55, -0.1)):
        angles = 2 * np.pi * np.arange(num_sides+1)/(num_sides)
        basic_vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        corners = np.asarray(center) + inscribed_radius * 0.5 * basic_vertices
        self.draw_points(corners)

    def control(self, desired_pose, ref_frame):
        """
        Adjust the robot so the object has a constant pose (target_pose) in the reference ref_frame
        Args:
            target_pose: <list> pose as [x,y,z,qw,qx,qy,qz]
            ref_frame: <str>
        """
        T_desired = tr.quaternion_matrix(desired_pose[3:])
        T_desired[:3, 3] = desired_pose[:3]
        try:
            while not rospy.is_shutdown():
                print('Control')
                # Read object position
                current_marker_pose = self.get_marker_pose()
                T_mf = tr.quaternion_matrix(current_marker_pose['pose'][3:])
                T_mf[:3,3] = current_marker_pose['pose'][:3]
                T_mf_desired = T_desired @ np.linalg.inv(T_mf)   # maybe it is this

                # Compute the target
                target_pose = np.concatenate([T_mf_desired[:3,3], tr.quaternion_from_matrix(T_mf_desired)])
                # broadcast target_pose:
                self.tf_broadcaster.sendTransform(list(target_pose[:3]), list(target_pose[3:]), rospy.Time.now(), '{}_desired'.format(current_marker_pose['frame']), ref_frame)
                self.tf_broadcaster.sendTransform(list(desired_pose[:3]), list(desired_pose[3:]), rospy.Time.now(), 'desired_obj_pose', ref_frame)
                self.tf_broadcaster.sendTransform(list(current_marker_pose['pose'][:3]), list(current_marker_pose['pose'][3:]), rospy.Time.now(), 'current_obj_pose', current_marker_pose['frame'])
                plan_result = self.med.plan_to_pose(self.med.arm_group, current_marker_pose['frame'], target_pose=list(target_pose), frame_id=ref_frame)
        except rospy.ROSInterruptException:
            pass


# TEST THE CODE: ------------------------------------------------------------------------------------------------------

def draw_test(supervision=False):
    bd = BubbleDrawer()
    # bd.draw_square()
    bd.draw_regular_polygon(3)
    bd.draw_regular_polygon(4)
    bd.draw_regular_polygon(5)
    bd.draw_regular_polygon(6)
    bd.draw_square()

if __name__ == '__main__':
    supervision = False
    draw_test(supervision=supervision)