#! /usr/bin/env python
import numpy as np

from bubble_control.bubble_drawer.bubble_drawer import BubbleDrawer

# TEST THE CODE: ------------------------------------------------------------------------------------------------------

def draw_test(supervision=False, reactive=False, adjust_lift=False):
    bd = BubbleDrawer(reactive=reactive, adjust_lift=adjust_lift)
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
    bd = BubbleDrawer(reactive=True, adjust_lift=False)
    center = (0.6, -0.25)
    center_2 = (0.6, 0.2)

    num_iters = 5

    for i in range(num_iters):
        bd.draw_regular_polygon(4, center=center, circumscribed_radius=0.15, init_angle=np.pi*0.25, end_raise=(i==num_iters-1))

    _ = input('Please, rearange the marker and press enter. ')
    bd.reactive = False
    for i in range(num_iters):
        bd.draw_regular_polygon(4, center=center_2, circumscribed_radius=0.15, init_angle=np.pi*0.25)


def reactive_pivoting_demo():
    bd = BubbleDrawer(reactive=True, adjust_lift=False)
    center = (0.6, -0.25)
    center_2 = (0.6, 0.2)
    num_iters = 5

    for i in range(num_iters):
        # bd.draw_regular_polygon(4, center=center, circumscribed_radius=0.15, init_angle=np.pi*0.25, end_raise=(i==num_iters-1))
        bd.draw_regular_polygon(4, center=center, circumscribed_radius=0.15, init_angle=np.pi * 0.25,
                                end_raise=(i == num_iters - 1), end_adjust=True, init_drawing=i == 0)

    _ = input('Please, rearange the marker and press enter. ')
    bd.reactive = False
    for i in range(num_iters):
        bd.draw_regular_polygon(4, center=center_2, circumscribed_radius=0.15, init_angle=np.pi * 0.25, init_drawing=i == 0, end_raise=(i == num_iters - 1), end_adjust=False)

def test_pivot():
    bd = BubbleDrawer(reactive=True)
    while True:
        _ = input('press enter to continue')
        bd.test_pivot_motion()

if __name__ == '__main__':
    supervision = False
    reactive = True
    # reactive = False
    adjust_lift = False

    # topic_recorder = TopicRecorder()
    # wrench_recorder = WrenchRecorder('/med/wrench', ref_frame='world')
    # wrench_recorder.record()
    # draw_test(supervision=supervision, reactive=reactive)
    # print('drawing done')
    # wrench_recorder.stop()
    # wrench_recorder.save('~/Desktop')

    # reactive_demo()
    reactive_pivoting_demo()
    # test_pivot()