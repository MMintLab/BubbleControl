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


class BubblePCReconstructor(object):

    def __init__(self, reconstruction_frame='grasp_frame', threshold=0.005, object_name='allen'):
        self.object_name = object_name
        self.reconstruction_frame = reconstruction_frame
        self.threshold = threshold
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

        # object_model = cylinder_pcd
        # object_model = planes_pcd
        # object_model = pen_pcd
        # object_model = spatula_pl_pcd
        object_model = marker_pcd
        # object_model = allen_pcd
        # object_model = paddle_pcd
        # TODO: Add rest
        models = {'allen': allen_pcd, 'marker': marker_pcd}
        object_model = models[self.object_name]

        return object_model

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

    def icp(self, threshold, view=False, verbose=False):
        imprint = self.get_imprint(view=view)
        # TODO: filter imprint to remove noise points
        imprint_mean = np.mean(imprint[:, :3], axis=0)
        dists = np.linalg.norm(imprint[:, :3] - imprint_mean, axis=1)
        d_th = 0.015
        imprint = imprint[np.where(dists<=d_th)]
        imprint_pcd = pack_o3d_pcd(imprint)
        imprint_mean = np.mean(imprint, axis=0)
        if self.last_tr is None:
            trans_init = np.eye(4)
            # trans_init[:3, 3] = imprint_mean[:3]#+0.01*np.std(imprint[:,:3], axis=0)*np.random.randn(3)
            _axis = np.random.uniform(-1, 1, 3)
            axis = _axis/np.linalg.norm(_axis)
            q_random = tr.quaternion_about_axis(np.random.uniform(-np.pi*0.1, np.pi*0.1), axis)
            T_random = tr.quaternion_matrix(q_random)
            # trans_init = T_random
            trans_init[:3, 3] = imprint_mean[:3]#+0.01*np.std(imprint[:,:3], axis=0)*np.random.randn(3)
        else:
            trans_init = self.last_tr
        if view:
            # visualize the initial transformation
            model_tr_pcd = copy.deepcopy(self.object_model)
            model_tr_pcd.transform(trans_init)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([imprint_pcd, model_tr_pcd, mesh_frame])
        icp_tr = self._icp(source_pcd=self.object_model, target_pcd=imprint_pcd, threshold=threshold, trans_init=trans_init, verbose=verbose)
        self.last_tr = icp_tr
        if view:
            # visualize the icp estimated transformation
            model_tr_pcd = copy.deepcopy(self.object_model)
            model_tr_pcd.transform(icp_tr)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([imprint_pcd, model_tr_pcd, mesh_frame])
        return icp_tr

    def _icp(self, source_pcd, target_pcd, threshold, trans_init, verbose=False):
        # Point-to-point:
        reg_p2p = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
        # point_to_plane:
        # > compute normals (required for point-to-plane icp)
        # imprint_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
        # reg_p2p = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
        if verbose:
            print(reg_p2p)
        icp_transformation = reg_p2p.transformation
        return icp_transformation

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


class BubblePoseEstimator(object):

    def __init__(self, imprint_th=0.005, icp_th=0.01, rate=1.0, view=False, verbose=False, object_name='allen'):
        self.object_name = object_name
        self.imprint_th = imprint_th
        self.icp_th = icp_th
        self.rate = rate
        self.view = view
        self.verbose = verbose
        rospy.init_node('bubble_pose_estimator')
        self.reconstructor = BubblePCReconstructor(threshold=self.imprint_th, object_name=self.object_name)
        self.marker_publisher = rospy.Publisher('estimated_object', Marker, queue_size=100)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.calibrate()
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
                icp_tr = self.reconstructor.icp(threshold=self.icp_th, view=self.view, verbose=verbose)
                self._broadcast_tr(icp_tr)
            except rospy.ROSInterruptException:
                break

    def _broadcast_tr(self, icp_tr):
        t = icp_tr[:3,3]
        q = tr.quaternion_from_matrix(icp_tr)
        marker_i = self._create_marker(t, q)
        self.marker_publisher.publish(marker_i)
        # add also the tf
        # self.tf_broadcaster.sendTransform(list(t), list(q), rospy.Time.now(), self.reconstructor.reconstruction_frame, 'tool_obj_frame')

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







