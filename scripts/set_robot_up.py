#! /usr/bin/env python
import numpy as np

import rospy
from arm_robots.med import Med

from victor_hardware_interface_msgs.msg import ControlMode, MotionStatus


def set_robot_up():
    home_joint_conf = [0, 0, 0, 0, 0, 0, 0]
    rospy.init_node('up_position')
    med = Med(display_goals=False)
    med.connect()
    med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.15)
    med.plan_to_joint_config(med.arm_group, home_joint_conf)


if __name__ == '__main__':
    set_robot_up()