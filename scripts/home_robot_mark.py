#! /usr/bin/env python
import numpy as np

import rospy
from arm_robots.med import Med

from victor_hardware_interface_msgs.msg import ControlMode, MotionStatus


def home_robot():
    home_joint_conf = [0, 1.0, 0, -0.8, 0, 0.9, 0] # mark home position
    # home_joint_conf = [0, 0.432, 0, -1.584, 0, 0.865, 0] # drawing home
    rospy.init_node('home_position')
    med = Med(display_goals=False)
    med.connect()
    med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.15)
    med.plan_to_joint_config(med.arm_group, home_joint_conf)


if __name__ == '__main__':
    home_robot()