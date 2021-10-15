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

from mmint_camera_utils.point_cloud_utils import pack_o3d_pcd, view_pointcloud
from mmint_camera_utils.point_cloud_parsers import PicoFlexxPointCloudParser
from bubble_control.bubble_pose_estimation.pose_estimators import ICP3DPoseEstimator, ICP2DPoseEstimator, ExplicitICP3DPoseEstimator


class BubblePCReconstructor(object):

    def __init__(self, reconstruction_frame='grasp_frame', threshold=0.005, object_name='allen', estimation_type='icp3d', view=False, path=None):
        self.object_name = object_name
        self.estimation_type = estimation_type
        self.reconstruction_frame = reconstruction_frame
        self.threshold = threshold
        self.view = view
        self.path = path
        self.left_parser = PicoFlexxPointCloudParser(camera_name='pico_flexx_left')
        self.right_parser = PicoFlexxPointCloudParser(camera_name='pico_flexx_right')
        self.reference_pcs = {
            'left_pc': None,
            'left_frame': None,
            'right_pc': None,
            'right_frame': None,
        }
        self.trees = {
            'right': None,
            'left': None,
        }
        self.radius = 0.005
        self.height = 0.12
        self.object_model = self._get_object_model()
        self.pose_estimator = self._get_pose_estimator()
        self.last_tr = None

    def _get_object_model(self):
        cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=self.radius*0.5, height=self.height*0.1, split=50)
        cylinder_pcd = o3d.geometry.PointCloud()
        cylinder_pcd.points = cylinder_mesh.vertices
        cylinder_pcd.paint_uniform_color([0, 0, 0])

        # object model simplified by 2 planes
        grid_y, grid_z = np.meshgrid(np.linspace(-self.radius*0.5, self.radius*0.5,10), np.linspace(-self.height*0.1, self.height*0.1, 30))
        points_y = grid_y.flatten()
        points_z = grid_z.flatten()
        plane_base = np.stack([np.zeros_like(points_y), points_y, points_z], axis=1)
        plane_base = np.concatenate([plane_base, np.zeros_like(plane_base)], axis=1)
        plane_1 = plane_base.copy()
        plane_2 = plane_base.copy()
        plane_1[:,0] = self.radius*1.2
        plane_2[:,0] = -self.radius*1.2
        planes_pc = np.concatenate([plane_1, plane_2], axis=0)
        planes_pcd = pack_o3d_pcd(planes_pc)

        # PEN ---- object model simplified by 2 planes
        grid_y, grid_z = np.meshgrid(np.linspace(-0.0025, 0.0025, 30),
                                     np.linspace(-self.height * 0.1, self.height * 0.1, 50))
        points_y = grid_y.flatten()
        points_z = grid_z.flatten()
        plane_base = np.stack([np.zeros_like(points_y), points_y, points_z], axis=1)
        plane_base = np.concatenate([plane_base, np.zeros_like(plane_base)], axis=1)
        plane_1 = plane_base.copy()
        plane_2 = plane_base.copy()
        plane_1[:, 0] = 0.006
        plane_2[:, 0] = -0.006
        pen_pc = np.concatenate([plane_1, plane_2], axis=0)
        pen_pcd = pack_o3d_pcd(pen_pc)

        # SPATULA ---- spatula simplified by 2 planes
        grid_y, grid_z = np.meshgrid(np.linspace(-0.0025, 0.0025, 50),
                                     np.linspace(-0.01, 0.01, 50))
        points_y = grid_y.flatten()
        points_z = grid_z.flatten()
        plane_base = np.stack([np.zeros_like(points_y), points_y, points_z], axis=1)
        plane_base = np.concatenate([plane_base, np.zeros_like(plane_base)], axis=1)
        plane_1 = plane_base.copy()
        plane_2 = plane_base.copy()
        plane_1[:, 0] = 0.009
        plane_2[:, 0] = -0.009
        spatula_pl_pc = np.concatenate([plane_1, plane_2], axis=0)
        spatula_pl_pcd = pack_o3d_pcd(spatula_pl_pc)

        # MARKER ---- object model simplified by 2 planes
        grid_y, grid_z = np.meshgrid(np.linspace(-0.0025, 0.0025, 30),
                                     np.linspace(-self.height * 0.1, self.height * 0.1, 50))
        points_y = grid_y.flatten()
        points_z = grid_z.flatten()
        plane_base = np.stack([np.zeros_like(points_y), points_y, points_z], axis=1)
        plane_base = np.concatenate([plane_base, np.zeros_like(plane_base)], axis=1)
        plane_1 = plane_base.copy()
        plane_2 = plane_base.copy()
        plane_1[:, 0] = 0.01
        plane_2[:, 0] = -0.01
        marker_pc = np.concatenate([plane_1, plane_2], axis=0)
        marker_pcd = pack_o3d_pcd(marker_pc)

        # ALLEN ---- object model simplified by 2 planes
        grid_y, grid_z = np.meshgrid(np.linspace(-0.0025, 0.0025, 30),
                                     np.linspace(-self.height * 0.1, self.height * 0.1, 50))
        points_y = grid_y.flatten()
        points_z = grid_z.flatten()
        plane_base = np.stack([np.zeros_like(points_y), points_y, points_z], axis=1)
        plane_base = np.concatenate([plane_base, np.zeros_like(plane_base)], axis=1)
        plane_1 = plane_base.copy()
        plane_2 = plane_base.copy()
        plane_1[:, 0] = 0.003
        plane_2[:, 0] = -0.003
        allen_pc = np.concatenate([plane_1, plane_2], axis=0)
        allen_pcd = pack_o3d_pcd(allen_pc)

        # PINGPONG PADDLE ---- object model simplified by 2 planes
        grid_y, grid_z = np.meshgrid(np.linspace(-0.01, 0.01, 30),
                                     np.linspace(-self.height * 0.1, self.height * 0.1, 50))
        points_y = grid_y.flatten()
        points_z = grid_z.flatten()
        plane_base = np.stack([np.zeros_like(points_y), points_y, points_z], axis=1)
        plane_base = np.concatenate([plane_base, np.zeros_like(plane_base)], axis=1)
        plane_1 = plane_base.copy()
        plane_2 = plane_base.copy()
        plane_1[:, 0] = 0.011
        plane_2[:, 0] = -0.011
        paddle_pc = np.concatenate([plane_1, plane_2], axis=0)
        paddle_pcd = pack_o3d_pcd(paddle_pc)

        if self.object_name == 'custom':
            custom_pc = o3d.io.read_point_cloud(self.path)
            custom_pcd = pack_o3d_pcd(custom_pc)        

        # object_model = cylinder_pcd
        # object_model = planes_pcd
        # object_model = pen_pcd
        # object_model = spatula_pl_pcd
        # object_model = marker_pcd
        # object_model = allen_pcd
        # object_model = paddle_pcd
        # TODO: Add rest
        models = {'allen': allen_pcd, 'marker': marker_pcd, 'pen': pen_pcd, 'custom': custom_pcd}
        object_model = models[self.object_name]

        return object_model       

    def _get_pose_estimator(self):
        pose_estimator = None
        available_esttimation_types = ['icp3d', 'icp2d', 'exp_icp3d']
        if self.estimation_type == 'icp3d':
            pose_estimator = ICP3DPoseEstimator(obj_model=self.object_model, view=self.view)
        elif self.estimation_type == 'icp2d':
            pose_estimator = ICP2DPoseEstimator(obj_model=self.object_model, projection_axis=(1,0,0), max_num_iterations=20)
        elif self.estimation_type == 'exp_icp3d':
            pose_estimator = ExplicitICP3DPoseEstimator(obj_model=self.object_model, max_num_iterations=20, view=self.view)
        else:
            raise NotImplementedError('pose estimation algorithm named "{}" not implemented yet. Available options: {}'.format(self.estimation_type, available_esttimation_types))
        return pose_estimator


    def reference(self):
        pc_r, frame_r = self.right_parser.get_point_cloud(return_ref_frame=True)
        pc_l, frame_l = self.left_parser.get_point_cloud(return_ref_frame=True)
        pc_r_filtered = self.filter_pc(pc_r)
        pc_l_filtered = self.filter_pc(pc_l)
        self.reference_pcs['left_pc'] = pc_l_filtered
        self.reference_pcs['right_pc'] = pc_r_filtered
        self.reference_pcs['left_frame'] = frame_l
        self.reference_pcs['right_frame'] = frame_r
        # Create the tree for improved performance
        self.trees['right'] = KDTree(self.reference_pcs['right_pc'][:,:3])
        self.trees['left'] = KDTree(self.reference_pcs['left_pc'][:,:3])
        self.last_tr = None

    def get_imprint(self, view=False):
        pc_r, frame_r = self.right_parser.get_point_cloud(return_ref_frame=True, ref_frame=self.reference_pcs['right_frame'])
        pc_l, frame_l = self.left_parser.get_point_cloud(return_ref_frame=True, ref_frame=self.reference_pcs['left_frame'])
        pc_r = self.filter_pc(pc_r)
        pc_l = self.filter_pc(pc_l)
        pc_r_contact_indxs = self._get_far_points_indxs(pc_r, d_threshold=self.threshold, key='right')
        pc_l_contact_indxs = self._get_far_points_indxs(pc_l, d_threshold=self.threshold, key='left')
        # pc_l_contact_indxs = get_far_points_indxs(self.reference_pcs['left_pc'], pc_l, d_threshold=self.threshold)
        pc_r_tr = self.right_parser.transform_pc(pc_r, origin_frame=frame_r, target_frame=self.reconstruction_frame)
        pc_l_tr = self.left_parser.transform_pc(pc_l, origin_frame=frame_l, target_frame=self.reconstruction_frame)
        # view
        pc_r_tr[:, 3] = 1 # paint it red
        pc_l_tr[:, 5] = 1 # paint it blue
        pc_r_tr[pc_r_contact_indxs, 3:6] =  np.array([0, 1, 0]) # green
        pc_l_tr[pc_l_contact_indxs, 3:6] =  np.array([0, 1, 0]) # green
        if view:
            print('visualizing the bubbles with the imprint on green')
            view_pointcloud([pc_r_tr, pc_l_tr], frame=True)

        imprint_r = pc_r_tr[pc_r_contact_indxs]
        imprint_l = pc_l_tr[pc_l_contact_indxs]
        if view:
            print('visualizing the imprint on green')
            view_pointcloud([imprint_r, imprint_l], frame=True)
        return np.concatenate([imprint_r, imprint_l], axis=0)

    def filter_pc(self, pc):
        # Fiter the raw pointcloud from the bubbles to remove the noisy limits. We obtain a kind of a cone
        angles = [10, -25, 20, -20]
        angles = [np.deg2rad(a) for a in angles]
        vectors = [np.array([0, 1, 0]), np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([1, 0, 0])]
        view_vector = np.array([0, 0, 1])
        conditions = []
        for angle_i, vector_i in zip(angles, vectors):
            q_i = tr.quaternion_about_axis(angle_i, axis=vector_i)
            R = tr.quaternion_matrix(q_i)[:3, :3]
            v_i = R @ view_vector
            q_perp = tr.quaternion_about_axis(-np.pi*0.5*np.sign(angle_i), axis=vector_i)
            R_perp = tr.quaternion_matrix(q_perp)[:3, :3]
            normal_i = R_perp @ v_i
            condition_i = np.dot(pc[:, :3], normal_i) >= 0
            conditions.append(condition_i)
        good_indxs = np.where(reduce((lambda x, y: x & y), conditions)) # aggregate all conditions
        filtered_pc = pc[good_indxs]
        return filtered_pc

    def estimate_pose(self, threshold, view=False, verbose=False):
        imprint = self.get_imprint(view=view)
        self.pose_estimator.threshold = threshold
        self.pose_estimator.verbose = verbose
        estimated_pose = self.pose_estimator.estimate_pose(imprint)
        return estimated_pose

    def _get_far_points_indxs(self, query_pc, d_threshold, key):
        """
        Compare the query_pc with the ref_pc and return the points in query_pc that are farther than d_threshold from ref_pc
        Args:
            ref_pc: <np.ndarray> (N,6)  reference point cloud
            query_pc: <np.ndarray> (N,6) query point cloud,
            d_threshold: <float> threshold distance to consider far if d>d_threshold
        Returns:
            - list of points indxs in query_pc that are far from ref_pc
        """
        tree = self.trees[key]
        if tree is None:
            print('tree not initialized yet')
            self.trees[key] = KDTree(self.reference_pcs['{}_pc'.format(key)][:, :3])
            tree = self.trees[key]
        qry_xyz = query_pc[:, :3]
        near_qry_indxs = tree.query_ball_point(qry_xyz, d_threshold)
        far_qry_indxs = [i for i, x in enumerate(near_qry_indxs) if len(x) == 0]
        return far_qry_indxs










