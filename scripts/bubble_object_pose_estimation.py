#!/usr/bin/env python3

import sys
import os
import rospy
import numpy as np
import ros_numpy as rn
import cv2
import ctypes
import struct
from PIL import Image as imm
import open3d as o3d
from scipy.spatial import KDTree
import copy
import tf
import tf.transformations as tr
from functools import reduce
from sklearn.cluster import DBSCAN
import argparse

import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from geometry_msgs.msg import TransformStamped, Pose
from visualization_msgs.msg import Marker, MarkerArray

from mmint_camera_utils.point_cloud_utils import *
from mmint_camera_utils.point_cloud_parsers import PicoFlexxPointCloudParser
from bubble_control.bubble_pose_estimation.bubble_pose_estimation import BubblePoseEstimator
from bubble_control.aux.load_confs import load_bubble_reconstruction_params


if __name__ == '__main__':
    # load params:
    params = load_bubble_reconstruction_params()
    object_names = list(params.keys())
    parser = argparse.ArgumentParser('Bubble Object Pose Estimation')
    parser.add_argument('object_name', type=str, help='Name of the object. Possible values: {}'.format(object_names))
    parser.add_argument('--reconstruction', type=str, default='tree', help='Name of imprint extraction algorithm. Possible values: (tree, depth)')
    parser.add_argument('--estimation_type', type=str, default='icp2d', help='Name of the algorithm used to estimate the pose from the imprint pc. Possible values: (icp2d, icp3d)')
    parser.add_argument('--rate', type=float, default=5.0, help='Estimated pose publishing rate (upper bound)')
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--verbose', action='store_true')


    args = parser.parse_args()

    object_name = args.object_name
    object_params = params[object_name]
    imprint_th = object_params['imprint_th'][args.reconstruction]
    icp_th = object_params['icp_th']
    gripper_width = object_params['gripper_width']

    print('-- Estimating the pose of a {} --'.format(object_name))
    bpe = BubblePoseEstimator(object_name=object_name,
                              imprint_th=imprint_th,
                              icp_th=icp_th,
                              rate=args.rate,
                              view=args.view,
                              verbose=args.verbose,
                              estimation_type=args.estimation_type,
                              reconstruction=args.reconstruction,
                              gripper_width=gripper_width)







