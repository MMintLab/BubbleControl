#! /usr/bin/env python
import os
import pdb
import sys
import numpy as np

import tf.transformations as tr


from bubble_control.bubble_drawer.bubble_drawer import BubbleDrawer


def draw_M(reactive=False, force_threshold=5.):
    bd = BubbleDrawer(reactive=reactive, force_threshold=force_threshold)
    m_points = np.load('/home/mmint/InstalledProjects/robot_stack/src/bubble_control/config/M.npy')
    m_points[:, 1] = m_points[:, 1]*(-1)
    # scale points:
    scale = 0.25
    corner_point = np.array([.75, .1])
    R = tr.quaternion_matrix(tr.quaternion_about_axis(angle=-np.pi*0.5, axis=np.array([0, 0, 1])))[:2, :2]
    m_points_rotated = m_points @ R.T
    m_points_scaled = corner_point + scale*m_points_rotated
    m_points_scaled = np.concatenate([m_points_scaled, m_points_scaled[0:1]], axis=0)
    bd.draw_points(m_points_scaled)

if __name__ == '__main__':
    supervision = False
    # reactive = True
    reactive = False

    draw_M(reactive=reactive, force_threshold=0.25)