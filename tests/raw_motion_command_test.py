#! /usr/bin/env python
import copy

import rospy
import numpy as np
from arm_robots.med import Med

from victor_hardware_interface_msgs.msg import MotionCommand


def create_motion_command_from_joint_values(joints_poses, joint_vels):
    mc = MotionCommand()
    mc.joint_position.joint_1 = joints_poses[0]
    mc.joint_position.joint_2 = joints_poses[1]
    mc.joint_position.joint_3 = joints_poses[2]
    mc.joint_position.joint_4 = joints_poses[3]
    mc.joint_position.joint_5 = joints_poses[4]
    mc.joint_position.joint_6 = joints_poses[5]
    mc.joint_position.joint_7 = joints_poses[6]
    mc.joint_velocity.joint_1 = joint_vels[0]
    mc.joint_velocity.joint_2 = joint_vels[1]
    mc.joint_velocity.joint_3 = joint_vels[2]
    mc.joint_velocity.joint_4 = joint_vels[3]
    mc.joint_velocity.joint_5 = joint_vels[4]
    mc.joint_velocity.joint_6 = joint_vels[5]
    mc.joint_velocity.joint_7 = joint_vels[6]
    mc.header.stamp = rospy.Time.now()
    return mc


def test_wrist_motion(rot_angle=0.1, num_steps=10, vel_magnitude=0.02, med=None):
    if med is None:
        med = Med()
        med.connect()

    current_joint_values = med.get_joint_positions(med.get_arm_joints())
    print('Current Joints:', [np.rad2deg(j) for j in current_joint_values])
    angle_step = rot_angle/num_steps
    for i in range(1,num_steps+1):
        joint_values = copy.deepcopy(current_joint_values)
        joint_values[-1] = joint_values[-1] + angle_step*i
        vels = [0.]*7
        if i <= num_steps:
            vels[-1] = vel_magnitude*np.sign(rot_angle)
        mc = create_motion_command_from_joint_values(joint_values, vels)
        med.arm_command_pub.publish(mc)
        rospy.sleep(0.1)


def send_joint_pose(joints, vels):
    med = Med()
    med.connect()
    mc = create_motion_command_from_joint_values(joints, vels)
    med.arm_command_pub.publish(mc)
    rospy.sleep(5.)


if __name__ == '__main__':
    rospy.init_node('raw_motion_command_test')
    med = Med()
    med.connect()
    test_wrist_motion(rot_angle=np.deg2rad(100), num_steps=1, vel_magnitude=0.1, med=med)
    test_wrist_motion(rot_angle=np.deg2rad(-100), num_steps=1, vel_magnitude=0.1, med=med)
    # send_joint_pose(joints=[0]*7, vels=[0]*7)


"""
# NOTES:
 - Commanded position out of limits:
     - SSR CompoundRequest is invalid violates limit: 
 - Velocity exceeding the maximum raises Internal Interpolation error.

"""
