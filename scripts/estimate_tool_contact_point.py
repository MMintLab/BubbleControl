#!/usr/bin/env python3
import rospy
import numpy as np
import argparse
from arc_utilities.tf2wrapper import TF2Wrapper
from bubble_control.bubble_contact_point_estimation.tool_contact_point_estimator import ToolContactPointEstimator
import tf.transformations as tr

def estimate_tool_contact_point(plane_pose=None):
    if plane_pose is not None:
        ref_frame = 'med_base'
        plane_frame_name = 'plane_frame'
        try:
            rospy.init_node('contact_point_estimator')
        except (rospy.exceptions.ROSInitException, rospy.exceptions.ROSException):
            pass
        tf2_listener = TF2Wrapper()
        # create the transform
        plane_pos, plane_quat = np.split(plane_pose, [3])
        tf2_listener.send_transform(plane_pos, plane_quat, ref_frame, plane_frame_name, is_static=True)
        tcpe = ToolContactPointEstimator(plane_frame=plane_frame_name)
    else:
        tcpe = ToolContactPointEstimator()


if __name__ == '__main__':
    default_pos = np.zeros(3)
    default_ori = np.array([0, 0, 0, 1])

    parser = argparse.ArgumentParser('Tool Contact Point Estimation')
    parser.add_argument('--pos', nargs=3, type=float, default=default_pos)
    parser.add_argument('--ori', nargs='+', type=float, default=default_ori)

    args = parser.parse_args()
    pos = np.asarray(args.pos)
    ori = np.asarray(args.ori)
    if len(ori) == 3:
        # euler angles (otherwise quaternion)
        ori = tr.quaternion_from_euler(ori[0], ori[1], ori[2])
    plane_pose = np.concatenate([pos, ori])
    print('Plane Pose:', plane_pose)
    estimate_tool_contact_point(plane_pose=plane_pose)