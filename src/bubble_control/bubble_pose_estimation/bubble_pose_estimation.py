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
import threading
import copy

import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2
from geometry_msgs.msg import TransformStamped, Pose
from visualization_msgs.msg import Marker, MarkerArray

from mmint_camera_utils.point_cloud_utils import *
from mmint_camera_utils.point_cloud_parsers import PicoFlexxPointCloudParser

from bubble_control.bubble_pose_estimation.bubble_pc_reconstruction import BubblePCReconstructor


class BubblePoseEstimator(object):

    def __init__(self, reconstruction_frame='grasp_frame', imprint_th=0.005, icp_th=0.01, rate=5.0, view=False, verbose=False, object_name='allen', estimation_type='icp3d'):
        self.object_name = object_name
        self.reconstruction_frame = reconstruction_frame
        self.imprint_th = imprint_th
        self.icp_th = icp_th
        self.rate = rate
        self.view = view
        self.verbose = verbose
        rospy.init_node('bubble_pose_estimator')
        self.reconstructor = BubblePCReconstructor(reconstruction_frame=self.reconstruction_frame, threshold=self.imprint_th, object_name=self.object_name, estimation_type=estimation_type, view=self.verbose)
        self.marker_publisher = rospy.Publisher('estimated_object', Marker, queue_size=100)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tool_estimated_pose = None
        self.alive = True
        self.calibrate()
        self.lock = threading.Lock()
        self.publisher_thread = threading.Thread(target=self._marker_publishing_loop)
        self.publisher_thread.start()
        self.estimate_pose(verbose=self.verbose)
        rospy.spin()

    def calibrate(self):
        _ = input('press enter to calibrate')
        self.reconstructor.reference()
        _ = input('calibration done, press enter to continue')

    def estimate_pose(self, verbose=False):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            try:
                icp_tr = self.reconstructor.estimate_pose(threshold=self.icp_th, view=self.view, verbose=verbose)
                with self.lock:
                    # update the tool_estimated_pose
                    t = icp_tr[:3, 3]
                    q = tr.quaternion_from_matrix(icp_tr)
                    self.tool_estimated_pose = np.concatenate([t, q])
            except rospy.ROSInterruptException:
                self.finish()
                break

    def _marker_publishing_loop(self):
        publish_rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            with self.lock:
                current_tool_pose = copy.deepcopy(self.tool_estimated_pose)
            if current_tool_pose is not None:
                marker_i = self._create_marker(current_tool_pose[:3], current_tool_pose[3:])
                self.marker_publisher.publish(marker_i)
            publish_rate.sleep()
            with self.lock:
                if not self.alive:
                    return

    def _create_marker(self, t, q):
        mk = Marker()
        mk.header.frame_id = self.reconstructor.reconstruction_frame
        mk.type = Marker.CYLINDER
        mk.scale.x = 2*self.reconstructor.radius
        mk.scale.y = 2*self.reconstructor.radius
        mk.scale.z = 2*self.reconstructor.height # make it larger
        # set color
        mk.color.r = 158/255.
        mk.color.g = 232/255.
        mk.color.b = 217/255.
        mk.color.a = 1.0
        # set position
        mk.pose.position.x = t[0]
        mk.pose.position.y = t[1]
        mk.pose.position.z = t[2]
        mk.pose.orientation.x = q[0]
        mk.pose.orientation.y = q[1]
        mk.pose.orientation.z = q[2]
        mk.pose.orientation.w = q[3]
        return mk

    def finish(self):
        with self.lock:
            self.alive = False
        self.publisher_thread.join()


if __name__ == '__main__':

    # Continuous  pose estimator:
    # view = False
    view = True
    # imprint_th = 0.0048 # for pen with gw 15
    # imprint_th = 0.0048 # for allen with gw 12
    imprint_th = 0.0053 # for marker with gw 20
    # imprint_th = 0.006 # for spatula with gripper width of 15mm
    icp_th = 1. # consider all points
    icp_th = 0.005 # for allen key

    bpe = BubblePoseEstimator(view=view, imprint_th=imprint_th, icp_th=icp_th, rate=5., verbose=view)