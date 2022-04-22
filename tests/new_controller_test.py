#! /usr/bin/env python
import numpy as np
import rospy
import time
import sys
from tqdm import tqdm


from bubble_utils.bubble_med.bubble_med import BubbleMed


def new_controller_test():
    """
    Keep moving the robot between a colsed set of motions.
    To test how long it takes for the controller to crash.
    Returns:
    """

    rospy.init_node('controller')

    med = BubbleMed(display_goals=False)

    desired_quat = np.array([-np.cos(np.pi / 4), np.cos(np.pi / 4), 0, 0])

    # square motion betwen 4 points on a plane parallel to xy plane
    _xs = np.linspace(0.45, 0.65, 2)
    _ys = np.linspace(-.3, .3, 2)
    _zs = [0.2]    # np.linspace(.1, .3, num_points_z)
    xs, ys, zs = np.meshgrid(_xs, _ys, _zs)
    positions = np.stack([
        [0.45, -.3, 0.2],
        [0.65, -.3, 0.2],
        [0.65, .3, 0.2],
        [0.45, .3, 0.2],
    ], axis=0)

    for iter in tqdm(range(10000)):
        for p_i, position_i in enumerate(positions):
            pose_i = np.concatenate([position_i, desired_quat], axis=-1)
            med.plan_to_pose(group_name=med.arm_group, ee_link_name='grasp_frame', target_pose=list(pose_i), frame_id='med_base')
            time.sleep(5.0)




if __name__ == '__main__':
    new_controller_test()


