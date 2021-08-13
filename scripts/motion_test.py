#! /usr/bin/env python
import os
import sys
import numpy as np
import rospy

from arm_robots.med import Med

from victor_hardware_interface.victor_utils import Stiffness
from victor_hardware_interface_msgs.msg import ControlMode, MotionStatus

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Vector3
from control_msgs.msg import FollowJointTrajectoryFeedback


def motion_test(supervision=True):
    rospy.init_node('motion_test')

    # Set up the robot
    med = Med(display_goals=False)
    med.connect()
    med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.1)

    # Wait for user to initialize the motions
    _ = input('Press enter to start')
    med.plan_to_joint_config(med.arm_group, [0, 0, 0, 0, 0, 0, 0])

    pose_1 = [0.903, -0.321, 0.298, 0.523, -0.523, 0.476, -0.476]
    pose_2 = [0.791, 0.413, 0.353, 0.523, -0.523, 0.476, -0.476]

    poses = [pose_1, pose_2, pose_1, pose_2]

    # med.set_control_mode(ControlMode.JOINT_IMPEDANCE, stiffness=Stiffness.STIFF, vel=0.075)  # Low vel for safety
    med.set_control_mode(ControlMode.JOINT_POSITION, vel=0.2)  # Low vel for safety
    for pose_i in poses:
        if supervision:
            med.set_execute(False)

        plan_result = med.plan_to_pose(med.arm_group, 'grasp_frame', target_pose=pose_i, frame_id="med_base")
        if supervision:
            for i in range(10):
                k = input('Execute plan (y: yes, r: replan, f: finish')
                if k == 'y':
                    med.set_execute(True)
                    med.follow_arms_joint_trajectory(plan_result.planning_result.plan.joint_trajectory)
                    break
                elif k == 'r':
                    break
                elif k == 'f':
                    return
                else:
                    pass








if __name__ == '__main__':
    supervision = False
    motion_test(supervision=supervision)