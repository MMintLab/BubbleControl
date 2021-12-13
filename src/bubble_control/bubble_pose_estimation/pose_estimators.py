import numpy as np
import abc
from mmint_camera_utils.point_cloud_utils import pack_o3d_pcd, view_pointcloud
import open3d as o3d
import copy
import tf.transformations as tr
from scipy.spatial import KDTree
from tqdm import tqdm
from mmint_utils.terminal_colors import term_colors
from mmint_camera_utils.ros_utils.publisher_wrapper import PublisherWrapper
from std_msgs.msg import Bool

class PCPoseEstimatorBase(abc.ABC):
    """
    Given an imprint of the infered object points and the model of the object, infer the object position
    """
    def __init__(self):
        super().__init__()
        self.tool_detected_publisher = PublisherWrapper(topic_name='tool_detected', msg_type=Bool)

    @abc.abstractmethod
    def estimate_pose(self, target_pc):
        pass


class ICPPoseEstimator(PCPoseEstimatorBase):
    def __init__(self, obj_model, view=False, verbose=False):
        super().__init__()
        self.object_model = obj_model
        self.last_tr = None
        self.threshold = None# TODO
        self.view = view
        self.verbose = verbose

    def estimate_pose(self, target_pc, init_tr=None):
        target_pc = self._filter_input_pc(target_pc)
        target_pcd = pack_o3d_pcd(target_pc)
        if init_tr is None:
            init_tr = self._get_init_tr(target_pcd)
        if self.view:
            # Visualize the initial transofrm:
            print('visualizing ICP initial configuration')
            model_tr_pcd = copy.deepcopy(self.object_model)
            # model_tr_pcd.transform(init_tr)
            view_pointcloud([target_pcd, model_tr_pcd], frame=True)

        # Estimate the transformation
        icp_tr = self._icp(source_pcd=self.object_model, target_pcd=target_pcd, threshold=self.threshold, init_tr=init_tr)

        if self.view:
            # Visualize the estimated transofrm:
            print('visualizing the ICP infered transformation')
            model_tr_pcd = copy.deepcopy(self.object_model)
            model_tr_pcd.transform(icp_tr)
            view_pointcloud([target_pcd, model_tr_pcd], frame=True)

        self.last_tr = icp_tr
        return icp_tr

    def _filter_input_pc(self, input_pc):
        return input_pc

    def _get_init_tr(self, target_pcd):
        if self.last_tr is None:
            init_tr = np.eye(4)
        else:
            init_tr =  self.last_tr
        return init_tr

    @abc.abstractmethod
    def _icp(self, source_pcd, target_pcd, threshold, init_tr):
        pass

    @abc.abstractmethod
    def _sample_random_tr(self):
        pass


class ICP3DPoseEstimator(ICPPoseEstimator):
    """
    Estimate the pose of the target_pc using Iterative Closest Points from Open3D
    """
    def _get_init_tr(self, target_pcd):
        if self.last_tr is None:
            init_tr = np.eye(4)
            # trans_init[:3, 3] = imprint_mean[:3]#+0.01*np.std(imprint[:,:3], axis=0)*np.random.randn(3)
            _axis = np.random.uniform(-1, 1, 3)
            axis = _axis / np.linalg.norm(_axis)
            q_random = tr.quaternion_about_axis(np.random.uniform(-np.pi * 0.1, np.pi * 0.1), axis)
            T_random = tr.quaternion_matrix(q_random)
            # trans_init = T_random
            init_tr[:3, 3] = np.mean(np.asarray(target_pcd.points))
        else:
            init_tr = self.last_tr
        return init_tr

    def _icp(self, source_pcd, target_pcd, threshold, init_tr):
        # Point-to-point:
        reg_p2p = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, threshold, init_tr,
                                                              o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                  max_iteration=1000))
        # point_to_plane:
        # > compute normals (required for point-to-plane icp)
        # imprint_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
        # reg_p2p = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
        if self.verbose:
            print(reg_p2p)
        icp_transformation = reg_p2p.transformation
        return icp_transformation

    def _filter_input_pc(self, input_pc):
        input_mean = np.mean(input_pc[:, :3], axis=0)
        dists = np.linalg.norm(input_pc[:, :3] - input_mean, axis=1)
        d_th = 0.015
        filtered_input = input_pc[np.where(dists <= d_th)]
        return filtered_input

    def _sample_random_tr(self):
        random_tr = tr.quaternion_matrix(tr.random_quaternion()) # no translation
        return random_tr


class ICP2DPoseEstimator(ICPPoseEstimator):
    """
    Constrain the ICP to be on a plane
    """
    def __init__(self, *args, projection_axis=(0,0,1), max_num_iterations=20, **kwargs):
        self.projection_axis = np.asarray(projection_axis)
        self.projection_tr = self._get_projection_tr()
        self.max_num_iterations = max_num_iterations
        super().__init__(*args, **kwargs)

    def _get_projection_tr(self):
        self.projection_axis = self.projection_axis/np.linalg.norm(self.projection_axis)
        z_axis = np.array([0, 0, 1])
        if np.all(self.projection_axis == z_axis):
            projection_tr = np.eye(4)
        else:
            rot_angle = np.arccos(np.dot(self.projection_axis, z_axis))
            _rot_axis = np.cross(z_axis, self.projection_axis)
            rot_axis = _rot_axis/np.linalg.norm(_rot_axis)
            projection_tr = tr.quaternion_matrix(tr.quaternion_about_axis(rot_angle, axis=rot_axis))
        return projection_tr

    def _project_pc(self, pc):
        projected_pc = pc @ self.projection_tr[:3, :3].T + self.projection_tr[:3, 3]
        return projected_pc

    def _unproject_pc(self, pc):
        unproject_tr = tr.inverse_matrix(self.projection_tr)
        unprojected_pc = pc @ unproject_tr[:3, :3].T + unproject_tr[:3, 3]
        return unprojected_pc

    def _get_init_tr(self, target_pcd):
        init_tr = np.eye(4)
        # get the mean of the target_pcd
        projected_target_points = self._project_pc(np.asarray(target_pcd.points))
        mean_position = np.mean(projected_target_points, axis=0)
        init_tr[:3, 3] = mean_position
        return init_tr

    def _sample_random_tr(self):
        random_angle = np.uniform(0, 2*np.pi)
        z_axis = np.array([0, 0, 1])
        random_tr = tr.quaternion_matrix(tr.quaternion_about_axis(random_angle, z_axis))
        return random_tr

    def _icp(self, source_pcd, target_pcd, threshold, init_tr):
        """

        Args:
            source_pcd: (model)
            target_pcd: (scene)
            treshold:
            init_tr:
        Returns:
        """
        icp_tr = init_tr
        source_points = self._project_pc(np.asarray(source_pcd.points))
        target_points = self._project_pc(np.asarray(target_pcd.points))
        if len(target_points) < 4:
            print(f"{term_colors.WARNING}Warning: No scene points provided{term_colors.ENDC}")
            self.tool_detected_publisher.data = False
            if self.last_tr is not None:
                return self.last_tr
            return init_tr
        else:
            self.tool_detected_publisher.data = True
        for i in range(self.max_num_iterations):
            # transform model
            source_tr = source_points @ icp_tr[:3, :3].T + icp_tr[:3, 3]

            # view:
            # if self.view:
            #     print('{}/{}'.format(i+1, self.max_num_iterations))
            #     model_tr_pcd = copy.deepcopy(self.object_model)
            #     model_tr_pcd.transform(icp_tr)
            #     view_pointcloud([target_pcd, model_tr_pcd], frame=True)

            # Estimate Correspondences
            tree = KDTree(source_tr[:, :2])
            corr_distances, cp_indxs = tree.query(target_points[:, :2])
            # cp_indxs = np.arange(len(target_points))
            # Apply correspondences
            source_points_corr = source_points[cp_indxs] # corresponed points in model to the scene points

            # Estimate transformation in 2D
            mu_m = np.mean(source_points_corr, axis=0) # model
            mu_s = np.mean(target_points, axis=0) # scene
            pm = source_points_corr - mu_m # model
            ps = target_points - mu_s # scene
            W = np.einsum('ij,ik->jk', ps[:, :2], pm[:, :2])
            rot_angle = np.arctan2(W[1,0] - W[0,1], np.sum(np.diag(W)))
            _new_icp_tr = tr.quaternion_matrix(tr.quaternion_about_axis(angle=rot_angle, axis=np.array([0,0,1])))
            R_star = _new_icp_tr[:2,:2]
            t_star = mu_s[:2] - R_star @ mu_m[:2]
            _new_icp_tr[:2,3] = t_star
            new_icp_tr = _new_icp_tr

            icp_tr = new_icp_tr
        if self.view:
            print('VIEW Fitted 2d pc on projected space')
            source_tr = source_points @ icp_tr[:3, :3].T + icp_tr[:3, 3]
            source_pc_tr = np.concatenate([source_tr, np.zeros((len(source_tr), 3))],axis=-1)
            source_pc_tr[:,3] = 1
            target_pc = np.concatenate([target_points, np.zeros((len(target_points), 3))],axis=-1)
            target_pc[:,4] = 1

            view_pointcloud([source_pc_tr, target_pc], frame=True)

        unproject_tr = tr.inverse_matrix(self.projection_tr)
        unprojected_icp_tr = unproject_tr @ icp_tr @ self.projection_tr

        if self.view:
            print('VIEW Fitted 2d pc UNPROJECTED space')
            model_tr_pcd = copy.deepcopy(self.object_model)
            model_tr_pcd.transform(unprojected_icp_tr)
            view_pointcloud([target_pcd, model_tr_pcd], frame=True)
        return unprojected_icp_tr


# Debug 2D version:
if __name__ == '__main__':
    # basic test with no projection required
    num_points = 50
    angles = np.linspace(0, 2 * np.pi, num_points)
    dist_deltas = .15*np.random.randn(num_points)
    _dists = 0.1*(np.ones_like(angles) + np.cumsum(dist_deltas))
    dists = np.stack([_dists]*2, axis=-1)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    model_color = np.zeros((num_points, 3))
    model_color[:, 0] = 1 # red
    obj_model_points = np.concatenate([circle*dists, np.zeros((num_points, 1))], axis=-1)
    obj_model = np.concatenate([obj_model_points, model_color], axis=-1)

    lims = np.array([-1, -.1, 0])
    obj_model_points = np.random.uniform(-lims, lims, (num_points,3))
    model_color = np.zeros((num_points, 3))
    model_color[:, 0] = 1  # red
    obj_model = np.concatenate([obj_model_points, model_color], axis=-1)

    true_translation = np.array([.5, .2, 0])
    true_angle = np.pi/4
    true_axis = np.array([0, 0, 1])
    true_tr = tr.quaternion_matrix(tr.quaternion_about_axis(angle=true_angle, axis=true_axis))
    true_tr[:3, 3] = true_translation

    scene_model_points = obj_model_points @ true_tr[:3, :3].T + true_tr[:3, 3]
    scene_model = np.concatenate([scene_model_points, np.zeros_like(scene_model_points)], axis=-1)
    scene_model[:, 3:] = np.array([0, 1, 0]) # color green
    projection_axis = np.array([0, 0, 1])

    split_in_two = True
    if split_in_two:
        half_num_points = int(num_points/2)
        scene_model[:half_num_points, 2] = 0.5
        scene_model[half_num_points:, 2] = -0.5

    object_model_pcd = pack_o3d_pcd(obj_model)
    icp_estimator = ICP2DPoseEstimator(obj_model=object_model_pcd, view=True, verbose=True, projection_axis=projection_axis)
    icp_estimator.max_num_iterations = 20
    pose = icp_estimator.estimate_pose(scene_model)

