#! /usr/bin/env python
import rospy
import numpy as np
import cv2
import os

from arc_utilities.tf2wrapper import TF2Wrapper
from mmint_camera_utils.camera_utils.camera_parsers import RealSenseCameraParser
from mmint_tools.camera_tools.img_utils import project_points_pinhole
from matplotlib import pyplot as plt
from scipy.spatial import KDTree


def transform_points(points, X):
    points_original_shape = points.shape
    points = points.reshape(-1,points_original_shape[-1]) # add batch_dim
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
    points_tr_h = points_h @ X.T
    points_tr = points_tr_h[...,:3]
    points_tr = points_tr.reshape(points_original_shape)
    return points_tr

def transform_vectors(vectors, X):
    vectors_original_shape = vectors.shape
    vectors = vectors.reshape(-1, vectors_original_shape[-1]) # add batch_dim
    vectors_tr = vectors @ X[:3,:3].T
    vectors_tr = vectors_tr.reshape(vectors_original_shape)
    return vectors_tr


def invert_img(img):
    origin_type = img.dtype
    img = img.astype(np.float32)
    max_v = np.max(img)
    min_v = np.min(img)
    # set max_v to min_v and min_v to max_v
    img_norm = (img-min_v)/(max_v-min_v)
    img_inv = img_norm*(min_v-max_v) + max_v
    img_inv = img_inv.astype(origin_type)
    return img_inv


if __name__ == '__main__':
    # Parameters to tune
    rospy.init_node('test_evaluation_drawing')
    tag_names = ['tag_5', 'tag_6', 'tag_7']
    camera_indx = 1
    board_x_size = 0.56
    board_y_size = 0.86
    tag_size = 0.09
    projected_img_size = (1000*np.array([board_x_size, board_y_size])).astype(np.int32) # u,v (x,y)

    tf_listener = TF2Wrapper()
    rsp = RealSenseCameraParser(camera_indx=camera_indx, verbose=False)
    camera_info_depth = rsp.get_camera_info_depth()
    camera_info_color = rsp.get_camera_info_color()
    camera_frame = 'camera_1_link'
    camera_optical_frame = 'camera_1_color_optical_frame' # TODO: replace with the one from camera_info
    tag_frames = ['{}_{}'.format(tag_name, camera_indx) for tag_name in tag_names]

    color_img_1 = rsp.get_image_color().copy()
    depth_img_1 = rsp.get_image_depth()

    w_X_cf = np.eye(4) # TODO: Replace with the true camera-to-world calibrated tf
    cf_X_cof = tf_listener.get_transform(parent=camera_frame, child=camera_optical_frame)
    w_X_cof = w_X_cf @ cf_X_cof
    cf_X_tags = [tf_listener.get_transform(parent=camera_frame, child=tag_frame) for tag_frame in tag_frames]
    cof_X_tags = [tf_listener.get_transform(parent=camera_optical_frame, child=tag_frame) for tag_frame in tag_frames]
    w_X_tags = [w_X_cf @ cf_X_tag for cf_X_tag in cf_X_tags]

    # estimate plane normal in w
    tag_poses_w = [X[:3,3] for X in w_X_tags]
    tag_plane_vectors_w = [pose_i - tag_poses_w[0] for pose_i in tag_poses_w[1:]]
    tag_plane_vectors_w = [pv/np.linalg.norm(pv) for pv in tag_plane_vectors_w]
    plane_normal_w = np.cross(tag_plane_vectors_w[1], tag_plane_vectors_w[0])


    # DO NOT TRUST THE ORIENTATION OF THE TAG SINCE IT CAN BE VERY NOISY
    # GET TF from world to board w_X_bc
    w_X_bc = np.eye(4)
    w_X_bc[:3,0] = tag_plane_vectors_w[0]
    w_X_bc[:3,2] = plane_normal_w
    w_X_bc[:3,1] = np.cross(plane_normal_w, tag_plane_vectors_w[0])
    w_X_bc[:3,3] = tag_poses_w[0]

    board_corners_bc = np.array([
        [-0.5*tag_size, 0.5*tag_size, 0],
        [board_x_size-0.5*tag_size, 0.5*tag_size, 0],
        [board_x_size-0.5*tag_size, 0.5*tag_size-board_y_size,0],
        [-0.5*tag_size, 0.5*tag_size-board_y_size, 0]
    ])
    board_corners_w = np.einsum('ij,kj->ki',w_X_bc[:3,:3],board_corners_bc) + w_X_bc[:3,3]
    cof_X_bc = np.linalg.inv(w_X_cof) @ w_X_bc
    board_corners_cof = np.einsum('ij,kj->ki', cof_X_bc[:3,:3],board_corners_bc) + cof_X_bc[:3,3]

    # tag_poses_w = np.stack(tag_poses_w, axis=0)
    # tag_poses_cof = transform_points(tag_poses_w, np.linalg.inv(w_X_cof))
    tag_poses_cof = np.stack([X[:3, 3] for X in cof_X_tags], axis=0)
    # board_corners_cof = np.append(board_corners_cof, tag_poses_cof, axis=0)



    # Get the image coordinates of the board corners
    board_corners_uvw = project_points_pinhole(board_corners_cof, camera_info_color['K'])
    tag_centers_uvw = project_points_pinhole(tag_poses_cof, camera_info_color['K'])
    board_corners_uv = np.floor(board_corners_uvw[..., :2]).astype(np.int32)
    tag_centers_uv = np.floor(tag_centers_uvw[..., :2]).astype(np.int32)
    print('Corners uv', board_corners_uv)

    axis_dirs = np.array([[0,1], [1,0], [0,-1], [-1,0]])
    axis = np.concatenate([np.zeros((1, 2), dtype=np.int32)]+[axis_dirs*(i+1) for i in range(10)], axis=0)
    uv_ext = np.concatenate([board_corners_uv + axis_i for axis_i in axis], axis=0)
    tag_uv_ext = np.concatenate([tag_centers_uv + axis_i for axis_i in axis], axis=0)

    detected_color_img_q = color_img_1.copy()
    detected_color_img_q[uv_ext[...,1], uv_ext[...,0]] = np.array([255,0,0]) # paint it red
    detected_color_img_q[tag_uv_ext[...,1], tag_uv_ext[...,0]] = np.array([0,255,0]) # paint it green
    # view image
    plt.figure(1)
    plt.imshow(detected_color_img_q)
    # plt.show()


    # wrap the points from prespective to plane
    # import pdb; pdb.set_trace()
    destination_uv = np.array([[0,0], [1,0], [1,1], [0,1]])*np.repeat(np.expand_dims(projected_img_size, axis=0), 4, axis=0)
    H, _ = cv2.findHomography(board_corners_uv, destination_uv, method=cv2.RANSAC, ransacReprojThreshold=3.0)

    unwarped_img = cv2.warpPerspective(color_img_1, H, tuple(projected_img_size), flags=cv2.INTER_LINEAR)

    plt.figure(2)
    plt.imshow(unwarped_img)

    #  PROCESS THE UNWARPED IMG
    processed_unwarped_img = unwarped_img.copy()
    # remove corners
    tag_pixel_size = int(1000*tag_size)
    board_pixel_x_size = int(1000*board_x_size)
    board_pixel_y_size = int(1000*board_y_size)

    def filter_corners(img):
        patch_size = int(1.1*tag_pixel_size)
        filtered_img = img.copy()
        max_value = np.max(img)
        filtered_img[:patch_size,:patch_size] = max_value
        filtered_img[:patch_size, board_pixel_x_size-patch_size:] = max_value
        filtered_img[board_pixel_y_size-patch_size:,:patch_size] = max_value
        return filtered_img


    gray_img = cv2.cvtColor(processed_unwarped_img, cv2.COLOR_BGR2GRAY)
    th, gray_img_th_otsu = cv2.threshold(gray_img, 128, 192, cv2.THRESH_OTSU)
    # import pdb; pdb.set_trace()
    gray_img_th_otsu_or = filter_corners(gray_img_th_otsu)
    gray_img_th_otsu = invert_img(gray_img_th_otsu_or)

    plt.figure(3)
    plt.imshow(gray_img_th_otsu)



    # Display points from board coordinates to rectified image coordinates
    num_points = 1000
    drawing_base = np.zeros_like(gray_img_th_otsu)
    drawing_board_coordinates_xs = board_x_size*0.5*np.ones((num_points,))-tag_size*0.5
    drawing_board_coordinates_ys = np.linspace(-board_y_size+0.5*tag_size,0.5*tag_size, num=num_points)
    drawing_board_coordinates_zs = np.zeros((num_points, ))
    drawing_bc = np.stack([drawing_board_coordinates_xs, drawing_board_coordinates_ys, drawing_board_coordinates_zs], axis=-1) # ub board coordinates
    drawing_cof = transform_points(drawing_bc, cof_X_bc) # on camera optical frame coordiantes
    drawing_uvs = project_points_pinhole(drawing_cof, camera_info_color['K'])[...,:2]
    drawing_uvs_rectified = np.clip(np.rint((np.concatenate([drawing_uvs, np.ones((drawing_uvs.shape[0], 1))],axis=-1) @ H.T)[...,:2]), np.zeros(2), np.flip(drawing_base.shape[:2])-1).astype(np.int32)

    drawing_base[drawing_uvs_rectified[...,1], drawing_uvs_rectified[...,0]] = 255 # paint it white


    # view desired drawing on the image
    drawing_uvs_int = np.clip(np.rint(drawing_uvs), np.zeros(2), np.flip(detected_color_img_q.shape[:2])-1).astype(np.int32)
    detected_color_img_q[drawing_uvs_int[...,1], drawing_uvs_int[...,0]] = np.array([255, 165,0]) # paint it orange

    plt.figure(4)
    plt.imshow(detected_color_img_q)

    plt.figure(5)
    plt.imshow(drawing_base)



    base_name = 'case_3'
    path = '~/Desktop/drawing_evaluation'
    # import pdb; pdb.set_trace()
    figs = [plt.figure(i) for i in plt.get_fignums()]
    for i, fig_num in enumerate(plt.get_fignums()):
        file_name = '{}_{}.png'.format(base_name, i)
        fig_path = os.path.join(path, file_name)
        if not os.path.exists(path):
            os.makedirs(path)
            print('created: ', path)
        # plt.figure(fig_num)
        plt.savefig(fig_path)


    plt.show(block=False)
    # Compute the score between the actual drawn image and the desired one.
    # -- For each point in the desired drawing, compute the closest one in the actual drawing


    actual_drawing = gray_img_th_otsu
    desired_drawing = drawing_base
    img_th = 50
    current_drawing_pixels = np.stack(np.where(actual_drawing>img_th), axis=-1)
    desired_drawing_pixels = np.stack(np.where(desired_drawing>img_th), axis=-1)
    tree = KDTree(current_drawing_pixels)
    min_dists, min_indxs = tree.query(desired_drawing_pixels)

    score = np.mean(min_dists)
    print('SCORE:', score)
    _ = input('Press enter')

    print('DONE')
