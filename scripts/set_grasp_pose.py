#! /usr/bin/env python
import numpy as np

import rospy
from arm_robots.med import Med

from victor_hardware_interface_msgs.msg import ControlMode, MotionStatus


def set_grasp_pose():
    grasp_pose_joints = [0.7613740469101997, 1.1146166859754167, -1.6834551714751782, -1.6882417308401203, 0.47044861033517205, 0.8857417788890095, 0.8497585444122142]
    rospy.init_node('grasp_pose')
    med = Med(display_goals=False)
    med.connect()
    med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.15)
    med.plan_to_joint_config(med.arm_group, grasp_pose_joints)


if __name__ == '__main__':
    set_grasp_pose()

