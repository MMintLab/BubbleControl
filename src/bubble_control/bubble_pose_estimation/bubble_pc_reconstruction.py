#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.spatial import KDTree
import tf.transformations as tr
import tf2_ros as tf2
from functools import reduce
import abc

import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, Bool
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped

from mmint_camera_utils.camera_utils.point_cloud_utils import pack_o3d_pcd, view_pointcloud, tr_pointcloud
from bubble_utils.bubble_parsers.bubble_parser import BubbleParser
from bubble_utils.bubble_tools.bubble_pc_tools import get_imprint_pc
from bubble_control.bubble_pose_estimation.pose_estimators import ICP3DPoseEstimator, ICP2DPoseEstimator
from mmint_camera_utils.ros_utils.publisher_wrapper import PublisherWrapper
from mmint_utils.terminal_colors import term_colors
from bubble_control.aux.load_confs import load_object_models


class BubblePCReconstructorBase(abc.ABC):
    """
    Gets Imprint and estimates the object pose from it
    """

    def __init__(self, reconstruction_frame='grasp_frame', threshold=0.005, percentile=None, object_name='allen', estimation_type='icp3d', view=False, verbose=False):
        self.object_name = object_name
        self.estimation_type = estimation_type
        self.reconstruction_frame = reconstruction_frame
        self.threshold = threshold
        self.percentile = percentile
        self.view = view
        self.verbose = verbose
        self.references = {
            'left': None,
            'left_frame': None,
            'right': None,
            'right_frame': None,
        }
        self.radius = 0.005
        self.height = 0.12
        self.object_model = self._get_object_model()
        self.pose_estimator = self._get_pose_estimator()
        self.tool_detected_publisher = PublisherWrapper(topic_name='tool_detected', msg_type=Bool)
        self.last_tr = None

    @abc.abstractmethod
    def reference(self):
        # save the reference state
        pass

    @abc.abstractmethod
    def get_imprint(self, view=False, separate=False):
        # return the contact imprint
        pass

    def _get_object_model(self):
        object_models = load_object_models()
        object_model = object_models[self.object_name]
        return object_model

    def _get_pose_estimator(self):
        pose_estimator = None
        available_esttimation_types = ['icp3d', 'icp2d']
        if self.estimation_type == 'icp3d':
            pose_estimator = ICP3DPoseEstimator(obj_model=self.object_model, view=self.view)
        elif self.estimation_type == 'icp2d':
            pose_estimator = ICP2DPoseEstimator(obj_model=self.object_model, projection_axis=(1,0,0), max_num_iterations=20, view=self.view)
        else:
            raise NotImplementedError('pose estimation algorithm named "{}" not implemented yet. Available options: {}'.format(self.estimation_type, available_esttimation_types))
        return pose_estimator

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

    def estimate_pose(self, threshold, view=False, verbose=False, tool_detection=True):
        if tool_detection:
            imprint, imprint_r, imprint_l = self.get_imprint(view=view, separate=True)
            imprint_pcd_r = pack_o3d_pcd(imprint_r)
            imprint_pcd_l = pack_o3d_pcd(imprint_l)
            distance_bubbles = None
            # Detect tool:
            if len(imprint_pcd_r.points) < 5 or len(imprint_pcd_l.points) < 5:
                # No tool detected
                if verbose:
                    print(f"{term_colors.WARNING}Warning: Not enough scene points provided (r: {len(imprint_pcd_r.points)}, l:{len(imprint_pcd_l.points)}){term_colors.ENDC}")
                self.tool_detected_publisher.data = False
            else:
                tree = KDTree(imprint_pcd_r.points)
                corr_distances, _ = tree.query(imprint_pcd_l.points)
                distance_bubbles = np.max(corr_distances)
                
                imprint_pcd_l.paint_uniform_color(np.array([1,0,0]))
                imprint_r_array = np.asarray(imprint_pcd_r.points)
                imprint_l_array = np.asarray(imprint_pcd_l.points)
                # Filter out outliers
                imprint_r_array = imprint_r_array[np.where(np.abs(imprint_r_array[:, 0]) < 0.02)]
                imprint_l_array = imprint_l_array[np.where(np.abs(imprint_l_array[:, 0]) < 0.02)]
                if imprint_r_array.shape[0] == 0 or imprint_l_array.shape[0] == 0:
                    if verbose:
                        print(f"{term_colors.WARNING}Warning: No scene points after filtering out outliers (r: {imprint_r_array.shape[0]}, l:{imprint_l_array.shape[0]}){term_colors.ENDC}")
                    self.tool_detected_publisher.data = False   
                else:
                    # Find the two points further apart in the x axis                 
                    x_max_r = np.max(imprint_r_array[:, 0])
                    x_min_l = np.min(imprint_l_array[:, 0])
                    distance_bubbles_x = np.abs(x_max_r - x_min_l)
                    
                    #view_pointcloud([imprint_pcd_l,imprint_pcd_r], frame=True)
                    if (distance_bubbles is not None and distance_bubbles_x < 0.01):
                        if verbose:
                            print(f"{term_colors.WARNING}Warning: No tool detected{term_colors.ENDC}")
                        self.tool_detected_publisher.data = False
                    else:
                        self.tool_detected_publisher.data = True
        else:
            imprint = self.get_imprint(view=view)
        estimated_pose = self._estimate_pose(imprint, threshold, verbose=verbose)
        return estimated_pose

    def _estimate_pose(self, imprint, threshold, verbose=False):
        self.pose_estimator.threshold = threshold
        self.pose_estimator.verbose = verbose
        estimated_pose = self.pose_estimator.estimate_pose(imprint)
        return estimated_pose


class BubblePCReconstructorROSBase(BubblePCReconstructorBase):
    """
    It adds the broadcasting and reading from ROS network.
    """

    def __init__(self, *args, broadcast_imprint=False, verbose=False, **kwargs):
        self.broadcast_imprint = broadcast_imprint
        self.verbose = verbose
        self.left_parser = BubbleParser(camera_name='pico_flexx_left', verbose=self.verbose)
        self.right_parser = BubbleParser(camera_name='pico_flexx_right', verbose=self.verbose)
        self.imprint_broadcaster = rospy.Publisher('imprint_pc', PointCloud2)
        super().__init__(*args, verbose=verbose, **kwargs)

    def _broadcast_imprint(self, imprint):
        header = Header()
        header.frame_id = self.reconstruction_frame
        xyz_points = imprint[:, :3].astype(np.float32)
        pc2_msg = pc2.create_cloud_xyz32(header, xyz_points)
        self.imprint_broadcaster.publish(pc2_msg)

    def _estimate_pose(self, imprint, threshold, verbose=False):
        if self.broadcast_imprint:
            self._broadcast_imprint(imprint)
        return super()._estimate_pose(imprint, threshold, verbose=verbose)


class BubblePCReconsturctorTreeSearch(BubblePCReconstructorROSBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trees = {
            'right': None,
            'left': None,
        }

    def reference(self):
        pc_r, frame_r = self.right_parser.get_point_cloud(return_ref_frame=True)
        pc_l, frame_l = self.left_parser.get_point_cloud(return_ref_frame=True)
        pc_r_filtered = self.filter_pc(pc_r)
        pc_l_filtered = self.filter_pc(pc_l)
        self.references['left'] = pc_l_filtered
        self.references['right'] = pc_r_filtered
        self.references['left_frame'] = frame_l
        self.references['right_frame'] = frame_r
        # Create the tree for improved performance
        self.trees['right'] = KDTree(self.references['right'][:, :3])
        self.trees['left'] = KDTree(self.references['left'][:, :3])
        self.last_tr = None

    def get_imprint(self, view=False, separate=False):
        pc_r, frame_r = self.right_parser.get_point_cloud(return_ref_frame=True, ref_frame=self.references['right_frame'])
        pc_l, frame_l = self.left_parser.get_point_cloud(return_ref_frame=True, ref_frame=self.references['left_frame'])
        pc_r = self.filter_pc(pc_r)
        pc_l = self.filter_pc(pc_l)
        pc_r_contact_indxs = self._get_far_points_indxs(pc_r, d_threshold=self.threshold, key='right')
        pc_l_contact_indxs = self._get_far_points_indxs(pc_l, d_threshold=self.threshold, key='left')
        # pc_l_contact_indxs = get_far_points_indxs(self.reference_pcs['left'], pc_l, d_threshold=self.threshold)
        pc_r_tr = self.right_parser.transform_pc(pc_r, origin_frame=frame_r, target_frame=self.reconstruction_frame)
        pc_l_tr = self.left_parser.transform_pc(pc_l, origin_frame=frame_l, target_frame=self.reconstruction_frame)
        # view
        pc_r_tr[:, 3] = 1 # paint it red
        pc_l_tr[:, 5] = 1 # paint it blue
        pc_r_tr[pc_r_contact_indxs, 3:6] = np.array([0, 1, 0]) # green
        pc_l_tr[pc_l_contact_indxs, 3:6] = np.array([0, 1, 0]) # green
        if view:
            print('visualizing the bubbles with the imprint on green')
            view_pointcloud([pc_r_tr, pc_l_tr], frame=True)

        imprint_r = pc_r_tr[pc_r_contact_indxs]
        imprint_l = pc_l_tr[pc_l_contact_indxs]
        if view:
            print('visualizing the imprint on green')
            view_pointcloud([imprint_r, imprint_l], frame=True)
        imprint = np.concatenate([imprint_r, imprint_l], axis=0)
        if separate:
            return imprint, imprint_r, imprint_l
        return imprint

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
            self.trees[key] = KDTree(self.references['{}_pc'.format(key)][:, :3])
            tree = self.trees[key]
        qry_xyz = query_pc[:, :3]
        near_qry_indxs = tree.query_ball_point(qry_xyz, d_threshold)
        far_qry_indxs = [i for i, x in enumerate(near_qry_indxs) if len(x) == 0]
        return far_qry_indxs


class BubblePCReconsturctorDepth(BubblePCReconstructorROSBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_info = {
            'left': self.left_parser.get_camera_info_depth(),
            'right': self.right_parser.get_camera_info_depth(),
        }

    def reference(self):
        # read refernce depth images
        depth_r = self.right_parser.get_image_depth()
        depth_l = self.left_parser.get_image_depth()
        self.references['left'] = depth_l
        self.references['right'] = depth_r
        self.references['left_frame'] = self.right_parser.optical_frame['depth']
        self.references['right_frame'] = self.left_parser.optical_frame['depth']
        self.last_tr = None

    def get_imprint(self, view=False, separate=False):
        depth_r = self.right_parser.get_image_depth()
        depth_l = self.left_parser.get_image_depth()
        imprint_r = get_imprint_pc(self.references['right'].squeeze(-1), depth_r.squeeze(-1), threshold=self.threshold, K=self.camera_info['right']['K'], percentile=self.percentile)
        imprint_l = get_imprint_pc(self.references['left'].squeeze(-1), depth_l.squeeze(-1), threshold=self.threshold, K=self.camera_info['left']['K'], percentile=self.percentile)
        frame_r = self.right_parser.optical_frame['depth']
        frame_l = self.left_parser.optical_frame['depth']

        filtered_imprint_r = self.filter_pc(imprint_r)
        filtered_imprint_l = self.filter_pc(imprint_l)
        
        # pc_l_contact_indxs = get_far_points_indxs(self.reference_pcs['left'], pc_l, d_threshold=self.threshold)
        imprint_r = self.right_parser.transform_pc(filtered_imprint_r, origin_frame=frame_r, target_frame=self.reconstruction_frame)
        imprint_l = self.left_parser.transform_pc(filtered_imprint_l, origin_frame=frame_l, target_frame=self.reconstruction_frame)
        
        # if view:
        #     pc_r_tr[pc_r_contact_indxs, 3:6] = np.array([0, 1, 0])  # green
        #     pc_l_tr[pc_l_contact_indxs, 3:6] = np.array([0, 1, 0])  # green
        #     print('visualizing the bubbles with the imprint on green')
        #     view_pointcloud([pc_r_tr, pc_l_tr], frame=True)
        if view:
            print('visualizing the imprint on green')
            # view
            imprint_r[:, 4] = 1  # paint it green
            imprint_l[:, 4] = 1  # paint it green
            view_pointcloud([imprint_r, imprint_l], frame=True)
        imprint = np.concatenate([imprint_r, imprint_l], axis=0)
        if separate:
            return imprint, imprint_r, imprint_l
        return imprint


class BubblePCReconstructorOfflineDepth(BubblePCReconstructorBase):
    def __init__(self, *args, **kwargs):
        self.depth_r = {
            'img': None,
            'frame': None,
        }
        self.depth_l = {
            'img': None,
            'frame': None,
        }
        self.camera_info = {
            'left': None,
            'right': None,
        }
        try:
            rospy.init_node('offline_reconstructor')
        except (rospy.exceptions.ROSInitException, rospy.exceptions.ROSException):
            pass

        self.buffer = tf2.BufferCore()

        super().__init__(*args, **kwargs)

    def reference(self):
        pass

    def add_tfs(self, tfs_df):
        for indx, row in tfs_df.iterrows():
            # pack the tf into a TrasformStamped message
            q_i = [row['qx'], row['qy'], row['qz'], row['qw']]
            t_i = [row['x'], row['y'], row['z']]
            parent_frame_id = row['parent_frame']
            child_frame_id = row['child_frame']
            ts_msg_i = self._pack_transform_stamped_msg(q_i, t_i, parent_frame_id=parent_frame_id, child_frame_id=child_frame_id)
            self.buffer.set_transform(ts_msg_i, 'default_authority')

    def _tr_pc(self, pc, origin_frame, target_frame):
        ts_msg = self.buffer.lookup_transform_core(target_frame, origin_frame, rospy.Time(0))
        t, R = self._unpack_transform_stamped_msg(ts_msg)
        pc_tr = tr_pointcloud(pc, R, t)
        return pc_tr

    def _unpack_transform_stamped_msg(self, ts_msg):
        x = ts_msg.transform.translation.x
        y = ts_msg.transform.translation.y
        z = ts_msg.transform.translation.z
        qx = ts_msg.transform.rotation.x
        qy = ts_msg.transform.rotation.y
        qz = ts_msg.transform.rotation.z
        qw = ts_msg.transform.rotation.w
        q = np.array([qx, qy, qz, qw])
        t = np.array([x, y, z])
        R = tr.quaternion_matrix(q)[:3,:3]
        return t, R

    def _pack_transform_stamped_msg(self, q, t, parent_frame_id, child_frame_id):
        ts_msg = TransformStamped()
        ts_msg.header.stamp = rospy.Time(0)
        ts_msg.header.frame_id = parent_frame_id
        ts_msg.child_frame_id = child_frame_id
        ts_msg.transform.translation.x = t[0]
        ts_msg.transform.translation.y = t[1]
        ts_msg.transform.translation.z = t[2]
        ts_msg.transform.rotation.x = q[0]
        ts_msg.transform.rotation.y = q[1]
        ts_msg.transform.rotation.z = q[2]
        ts_msg.transform.rotation.w = q[3]
        return ts_msg

    def get_imprint(self, view=False, separate=False):
        # return the contact imprint
        depth_r = self.depth_r['img']
        depth_l = self.depth_l['img']
        imprint_r = get_imprint_pc(self.references['right'].squeeze(-1), depth_r.squeeze(-1), threshold=self.threshold,
                                   K=self.camera_info['right']['K'], percentile=self.percentile)
        imprint_l = get_imprint_pc(self.references['left'].squeeze(-1), depth_l.squeeze(-1), threshold=self.threshold,
                                   K=self.camera_info['left']['K'], percentile=self.percentile)
        frame_r = self.depth_r['frame']
        frame_l = self.depth_l['frame']

        filtered_imprint_r = self.filter_pc(imprint_r)
        filtered_imprint_l = self.filter_pc(imprint_l)

        # trasform imprints
        imprint_r = self._tr_pc(filtered_imprint_r, frame_r, self.reconstruction_frame)
        imprint_l = self._tr_pc(filtered_imprint_l, frame_l, self.reconstruction_frame)

        if view:
            print('visualizing the imprint on green')
            # view
            imprint_r[:, 3] = 1  # paint it red
            imprint_l[:, 4] = 1  # paint it green
            view_pointcloud([imprint_r, imprint_l], frame=True)

        imprint = np.concatenate([imprint_r, imprint_l], axis=0)
        if separate:
            return imprint, imprint_r, imprint_l
        return imprint