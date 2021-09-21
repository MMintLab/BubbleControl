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

from bubble_control.bubble_data_collection.bubble_draw_data_collection import BubbleDrawingDataCollection

# TEST THE CODE: ------------------------------------------------------------------------------------------------------


def collect_data_drawing_test(supervision=False, reactive=False):

    save_path = '/home/mmint/Desktop/drawing_data'
    scene_name = 'drawing_data_test'

    dc = BubbleDrawingDataCollection(data_path=save_path, scene_name=scene_name, supervision=True)
    dc.collect_data(num_data=10)


if __name__ == '__main__':
    collect_data_drawing_test()