#! /usr/bin/env python
import os
import pdb
import sys
import numpy as np
import threading
import copy
import rospy
import tf
import tf.transformations as tr

from bubble_control.bubble_drawer.bubble_drawer import BubbleDrawer

# TEST THE CODE: ------------------------------------------------------------------------------------------------------

def draw_test(supervision=False, reactive=False):
    bd = BubbleDrawer(reactive=reactive)
    # center = (0.55, -0.25) # good one
    center = (0.45, -0.25)
    # center_2 = (0.55, 0.2)
    center_2 = (0.45, 0.2)

    # bd.draw_square()
    # bd.draw_regular_polygon(3, center=center)
    # bd.draw_regular_polygon(4, center=center)
    # bd.draw_regular_polygon(5, center=center)
    # bd.draw_regular_polygon(6, center=center)
    for i in range(5):
        # bd.draw_square(center=center_2)
        bd.draw_regular_polygon(3, center=center, circumscribed_radius=0.15)
    # bd.draw_square(center=center, step_size=0.04)


    # bd.draw_square(center=center_2)
    # bd.draw_square(center=center_2)
    # bd.draw_square(center=center_2)
    # bd.draw_square(center=center_2, step_size=0.04)

    # bd.draw_circle()

def reactive_demo():
    bd = BubbleDrawer(reactive=True)
    center = (0.6, -0.25)
    center_2 = (0.6, 0.2)

    num_iters = 5

    for i in range(num_iters):
        bd.draw_regular_polygon(4, center=center, circumscribed_radius=0.15, init_angle=np.pi*0.25)

    _ = input('Please, rearange the marker and press enter. ')
    bd.reactive = False
    for i in range(num_iters):
        bd.draw_regular_polygon(4, center=center_2, circumscribed_radius=0.15, init_angle=np.pi*0.25)

def test_pivot():
    bd = BubbleDrawer(reactive=True)
    while True:
        _ = input('press enter to continue')
        bd.test_pivot_motion()

if __name__ == '__main__':
    supervision = False
    reactive = True
    # reactive = False

    from mmint_camera_utils.topic_recording import TopicRecorder, WrenchRecorder

    # topic_recorder = TopicRecorder()
    # wrench_recorder = WrenchRecorder('/med/wrench', ref_frame='world')
    # wrench_recorder.record()
    # draw_test(supervision=supervision, reactive=reactive)
    # print('drawing done')
    # wrench_recorder.stop()
    # wrench_recorder.save('~/Desktop')

    reactive_demo()
    # test_pivot()