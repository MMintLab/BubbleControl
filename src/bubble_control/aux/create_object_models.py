import numpy as np
from mmint_camera_utils.point_cloud_utils import pack_o3d_pcd, view_pointcloud, tr_pointcloud
import open3d as o3d

from bubble_control.aux.load_confs import save_object_models


def create_object_models(radius=0.005, height=0.12):
    cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius * 0.5, height=height * 0.1,
                                                              split=50)
    cylinder_pcd = o3d.geometry.PointCloud()
    cylinder_pcd.points = cylinder_mesh.vertices
    cylinder_pcd.paint_uniform_color([0, 0, 0])

    # object model simplified by 2 planes
    grid_y, grid_z = np.meshgrid(np.linspace(-radius * 0.5, radius * 0.5, 10),
                                 np.linspace(-height * 0.1, height * 0.1, 30))
    points_y = grid_y.flatten()
    points_z = grid_z.flatten()
    plane_base = np.stack([np.zeros_like(points_y), points_y, points_z], axis=1)
    plane_base = np.concatenate([plane_base, np.zeros_like(plane_base)], axis=1)
    plane_1 = plane_base.copy()
    plane_2 = plane_base.copy()
    plane_1[:, 0] = radius * 1.2
    plane_2[:, 0] = -radius * 1.2
    planes_pc = np.concatenate([plane_1, plane_2], axis=0)
    planes_pcd = pack_o3d_pcd(planes_pc)

    # PEN ---- object model simplified by 2 planes
    grid_y, grid_z = np.meshgrid(np.linspace(-0.0025, 0.0025, 30),
                                 np.linspace(-height * 0.1, height * 0.1, 50))
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
                                 np.linspace(-height * 0.1, height * 0.1, 50))
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
                                 np.linspace(-height * 0.1, height * 0.1, 50))
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
                                 np.linspace(-height * 0.1, height * 0.1, 50))
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
    # object_model = marker_pcd
    # object_model = allen_pcd
    # object_model = paddle_pcd
    # TODO: Add rest
    models = {'allen': allen_pcd, 'marker': marker_pcd, 'pen': pen_pcd}

    return models


# Save them:

if __name__ == '__main__':
    radius = 0.005
    height = 0.12
    models = create_object_models(radius=radius, height=height)
    save_object_models(models)