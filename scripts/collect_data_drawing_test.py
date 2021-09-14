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
from mmint_camera_utils.topic_recording import TopicRecorder, WrenchRecorder
from mmint_camera_utils.point_cloud_parsers import PicoFlexxPointCloudParser

# TEST THE CODE: ------------------------------------------------------------------------------------------------------


def collect_data_drawing_test(supervision=False, reactive=False):

    save_path = '/home/mmint/Desktop'
    scene_name = 'drawing_data_test'

    wrench_recorder = WrenchRecorder('/med/wrench', ref_frame='world')
    bd = BubbleDrawer(reactive=reactive)

    camera_name_right = 'pico_flexx_right'
    camera_name_left = 'pico_flexx_left'
    camera_parser_right = PicoFlexxPointCloudParser(camera_name=camera_name_right, scene_name=scene_name)
    camera_parser_left = PicoFlexxPointCloudParser(camera_name=camera_name_left, scene_name=scene_name)

    num_data = 3

    for i in range(num_data):
        drawing_area_center_point = np.array([0.55, 0.])
        drawing_area_size = np.array([.1, .1])
        start_point_i = np.random.uniform(drawing_area_center_point-drawing_area_size, drawing_area_center_point+drawing_area_size, (2,))
        direction_i = np.random.uniform(0, 2*np.pi) # assume planar motion only
        length_i = np.random.uniform(0.01, 0.1)
        end_point_i = start_point_i + length_i * np.array([np.cos(direction_i), np.sin(direction_i)])
        # TODO: check that both points are inside the limits
        drawing_points_i = np.stack([start_point_i, end_point_i])

        # Start recording
        wrench_recorder.record()
        # Draw ----
        bd.draw_points(drawing_points_i)
        wrench_recorder.stop()
        wrench_save_path = ''
        wrench_recorder.save(, filename='wrench_{}'.format(i))
        wrench_recorder.reset()




if __name__ == '__main__':

    collect_data_drawing_test()