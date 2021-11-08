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
import argparse

from bubble_control.bubble_data_collection.bubble_draw_data_collection import BubbleDrawingDataCollection

# TEST THE CODE: ------------------------------------------------------------------------------------------------------


def collect_data_drawing_test(save_path, scene_name, num_data=10, prob_axis=0.08):

    dc = BubbleDrawingDataCollection(data_path=save_path, scene_name=scene_name, prob_axis=prob_axis)
    dc.collect_data(num_data=num_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Collect Data Drawing')
    parser.add_argument('save_path', type=str, help='path to save the data')
    parser.add_argument('num_data', type=int, help='Number of data samples to be collected')
    parser.add_argument('--scene_name', type=str, default='drawing_data', help='scene name for the data. For organization purposes')
    parser.add_argument('--prob_axis', type=float, default=0.08, help='probability for biasing the drawing along the axis')

    args = parser.parse_args()

    save_path = args.save_path
    scene_name = args.scene_name
    num_data = args.num_data
    prob_axis = args.prob_axis


    collect_data_drawing_test(save_path, scene_name, num_data=num_data, prob_axis=prob_axis)