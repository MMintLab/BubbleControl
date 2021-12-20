#! /usr/bin/env python
import os
import pdb
import sys
import numpy as np

import tf.transformations as tr

from geometry_msgs.msg import Pose, Point, Quaternion

from bubble_control.bubble_drawer.bubble_drawer import BubbleDrawer
from victor_hardware_interface_msgs.msg import ControlMode


def get_euler_from_quat(quat):
    euler_rad = np.asarray(tr.euler_from_quaternion(quat))
    euler_deg = np.rad2deg(euler_rad)
    return euler_deg

def draw_test(reactive=False, force_threshold=5.):
    bd = BubbleDrawer(reactive=reactive, force_threshold=force_threshold)
    bd.set_robot_conf('home_conf')
    current_pose = bd.get_current_pose('grasp_frame')
    desired_pose = current_pose.copy()
    desired_pose[3:] = bd.draw_quat
    bd.plan_to_pose(bd.arm_group, 'grasp_frame', list(desired_pose), frame_id='med_base')
    bd.set_control_mode(control_mode=ControlMode.CARTESIAN_IMPEDANCE, vel=0.25)

    def delta_cartesian_move(dx, dy, quat=None, quat_frame='grasp_frame'):
        if quat is None:
            reached = bd.move_delta_cartesian_impedance(arm=0, dx=dx, dy=dy)
        else:
            ee_pose_gf = bd.get_current_pose(quat_frame,
                                             ref_frame='med_kuka_link_ee')  # IMPEDANCE ORIENTATION IN MED_KUKA_LINK_EE FRAME!
            # desired_quat = tr.quaternion_multiply(ee_pose_gf[3:], quat)
            desired_quat = tr.quaternion_multiply(quat, tr.quaternion_inverse(ee_pose_gf[3:]))
            target_orientation = Quaternion()
            target_orientation.x = desired_quat[0]
            target_orientation.y = desired_quat[1]
            target_orientation.z = desired_quat[2]
            target_orientation.w = desired_quat[3]
            reached = bd.move_delta_cartesian_impedance(arm=0, dx=dx, dy=dy, target_orientation=target_orientation)
        return reached

    # Move:
    reached = delta_cartesian_move(0.1, 0)
    print('REACHED: ', reached)
    reached = delta_cartesian_move(dx=-0.1, dy=0, quat=bd.draw_quat)
    print('REACHED: ', reached)
    current_quat = bd.get_current_pose(frame_id='grasp_frame')[3:]
    for i in range(10):
        deg_angle = np.random.choice([-5,5])
        rot_quat = tr.quaternion_about_axis(angle=deg_angle*np.pi/180, axis=(1, 0, 0))
        new_quat = tr.quaternion_multiply(current_quat, rot_quat)
        reached = delta_cartesian_move(dx=0.01, dy=0, quat=new_quat)
        print('{} - REACHED: {}'.format(i, reached))
        current_quat = new_quat

if __name__ == '__main__':
    draw_test()