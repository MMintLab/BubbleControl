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
from wsg_50_utils.wsg_50_gripper import WSG50Gripper

from bubble_control.bubble_pose_estimation.bubble_pc_reconstruction import BubblePCReconsturctorDepth, BubblePCReconsturctorTreeSearch


class BubblePoseEstimator(object):
    """
    BubblePoseEstimation > BubblePCReconstructor > PoseEstimators
    """

    def __init__(self, imprint_th=0.005, icp_th=0.01, rate=5.0, percentile=None, view=False, verbose=False, broadcast_imprint=False, object_name='allen', estimation_type='icp3d', reconstruction='depth', gripper_width=None):
        self.object_name = object_name
        self.imprint_th = imprint_th
        self.icp_th = icp_th
        self.rate = rate
        self.percentile = percentile
        self.view = view
        self.verbose = verbose
        self.broadcast_imprint = broadcast_imprint
        self.estimation_type = estimation_type
        self.gripper_width = gripper_width
        try:
            rospy.init_node('bubble_pose_estimator')
        except (rospy.exceptions.ROSInitException, rospy.exceptions.ROSException):
            pass
        self.gripper = WSG50Gripper()
        self.reconstructor = self._get_reconstructor(reconstruction)
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

    def _get_reconstructor(self, reconstruction_key):
        reconstructors = {
            'depth': BubblePCReconsturctorDepth,
            'tree': BubblePCReconsturctorTreeSearch,
        }
        if reconstruction_key not in reconstructors:
            raise KeyError('No reconstructor found for key {} -- Possible keys: {}'.format(reconstruction_key, reconstructors.keys()))
        Reconstructor = reconstructors[reconstruction_key]
        reconstructor = Reconstructor(threshold=self.imprint_th, object_name=self.object_name, estimation_type=self.estimation_type,
                              view=self.view, verbose=self.verbose, broadcast_imprint=self.broadcast_imprint, percentile=self.percentile)
        return reconstructor

    def calibrate(self):
        info_msg = 'Press enter to calibrate --'
        if self.gripper_width is not None:
            info_msg += '\n\t>>> We will open the gripper!\t'
        _ = input(info_msg)
        if self.gripper_width is not None:
            # Open the gripper
            self.gripper.open_gripper()
        self.reconstructor.reference()
        info_msg = 'Calibration done! {}\nPress enter to continue :)'
        additional_msg = ''
        if self.gripper_width is not None:
            additional_msg = '\n We will close the gripper to a width {}mm'.format(self.gripper_width)
        _ = input(info_msg.format(additional_msg))
        if self.gripper_width is not None:
            # move gripper to gripper_width
            self.gripper.move(self.gripper_width, speed=50.0)

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
            rate.sleep()

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