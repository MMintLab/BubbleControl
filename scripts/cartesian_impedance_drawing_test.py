#! /usr/bin/env python
import os
import pdb
import sys
import numpy as np

import mmint_tools.mmint_tf_tools.transformations as tr

from geometry_msgs.msg import Pose, Point, Quaternion

from bubble_drawing.bubble_drawer.bubble_drawer import BubbleDrawer
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
    
    
    def delta_cartesian_move(dx, dy, target_z=None, quat=None, frame_id='grasp_frame'):
        # TODO: There is a small distance between the med_kuk_link_ee frame and the one used in computations (like get position). We could fix that, but empirically the overall the error seems small
        arm_id = 0
        target_orientation = None
        cartesian_motion_frame_id = 'med_kuka_link_ee'
        if quat is not None:
            # Find the transformation to the med_kuka_link_ee --- Cartesian Impedance mode sets the pose with respect to the med_kuka_link_ee frame
            ee_pose_gf = bd.get_current_pose(frame_id,
                                                   ref_frame=cartesian_motion_frame_id)  # IMPEDANCE ORIENTATION IN MED_KUKA_LINK_EE FRAME!
            # desired_quat = tr.quaternion_multiply(ee_pose_gf[3:], quat)
            desired_quat = tr.quaternion_multiply(quat, tr.quaternion_inverse(ee_pose_gf[3:]))
            target_orientation = Quaternion()
            target_orientation.x = desired_quat[0]
            target_orientation.y = desired_quat[1]
            target_orientation.z = desired_quat[2]
            target_orientation.w = desired_quat[3]

        wf_pose_gf = bd.get_current_pose(frame_id, ref_frame=bd.cartesian.sensor_frames[arm_id])
        wf_pose_ee = bd.get_current_pose(cartesian_motion_frame_id,
                                               ref_frame=bd.cartesian.sensor_frames[arm_id])
        ee_pose_gf = bd.get_current_pose(ref_frame=cartesian_motion_frame_id,
                                               frame_id=frame_id)  # this should be constant

        wf_pose_gf_desired = wf_pose_gf.copy()
        # wf_pose_ee_desired = wf_pose_gf.copy()
        wf_pose_gf_desired[:2] = wf_pose_gf_desired[:2] + np.array([dx, dy])
        if quat is not None:
            wf_pose_gf_desired[3:] = desired_quat

        wf_pose_ee_desired = bd._matrix_to_pose(bd._pose_to_matrix(wf_pose_gf_desired) @ np.linalg.inv(bd._pose_to_matrix(ee_pose_gf)))
        wf_pose_ee_delta = wf_pose_ee_desired - wf_pose_ee
        dxdy = wf_pose_ee_delta[:2]
        dx_desired, dy_desired = dxdy
        # import pdb; pdb.set_trace()

        if target_z is not None:
            delta_z = wf_pose_ee[2] - wf_pose_gf[2]
            # delta_z = wf_pose_ee_desired[2] = wf_pose_gf_desired[2]
            target_z = target_z + delta_z
        reached = bd.move_delta_cartesian_impedance(arm=arm_id, dx=dx_desired, dy=dy_desired, target_z=target_z, target_orientation=target_orientation) # positions with respect the ee_link
        return reached

    # Move:
    # reached = delta_cartesian_move(0.1, 0)
    # print('REACHED: ', reached)
    # reached = delta_cartesian_move(dx=-0.1, dy=0, quat=bd.draw_quat)
    # print('REACHED: ', reached)
    # current_quat = bd.get_current_pose(frame_id='grasp_frame')[3:]
    # for i in range(10):
    #     deg_angle = np.random.choice([-5,5])
    #     rot_quat = tr.quaternion_about_axis(angle=deg_angle*np.pi/180, axis=(1, 0, 0))
    #     new_quat = tr.quaternion_multiply(current_quat, rot_quat)
    #     reached = delta_cartesian_move(dx=0.01, dy=0, quat=new_quat)
    #     print('{} - REACHED: {}'.format(i, reached))
    #     current_quat = new_quat

    # test rotation
    current_pose = bd.get_current_pose(frame_id='grasp_frame')
    current_quat = current_pose[3:]
    deg_angle = 3
    desired_z = current_pose[2]
    for i in range(10):
        rot_quat = tr.quaternion_about_axis(angle=deg_angle*np.pi/180, axis=(1, 0, 0))
        new_quat = tr.quaternion_multiply(current_quat, rot_quat)
        reached = delta_cartesian_move(dx=0.02, dy=0, quat=new_quat, target_z=desired_z)
        # reached = delta_cartesian_move(dx=0.01, dy=0, quat=None, target_z=desired_z)
        print('{} - REACHED: {}'.format(i, reached))
        current_quat = new_quat

if __name__ == '__main__':
    draw_test()