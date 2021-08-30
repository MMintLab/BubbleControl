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

import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from geometry_msgs.msg import TransformStamped, Pose
from visualization_msgs.msg import Marker, MarkerArray

from mmint_camera_utils.point_cloud_utils import *
from mmint_camera_utils.point_cloud_parsers import PicoFlexxPointCloudParser
from bubble_control.bubble_pc_reconstruction import BubblePoseEstimator


if __name__ == '__main__':

    # Continuous  pose estimator:
    view = False
    # view = True
    # imprint_th = 0.0048 # for pen with gw 15
    # imprint_th = 0.0048 # for allen with gw 12
    imprint_th = 0.0053 # for marker with gw 20
    # imprint_th = 0.006 # for spatula with gripper width of 15mm
    icp_th = 1. # consider all points
    icp_th = 0.005 # for allen key
    bpe = BubblePoseEstimator(view=view, imprint_th=imprint_th, icp_th=icp_th, rate=5., verbose=view, object_name='marker')







