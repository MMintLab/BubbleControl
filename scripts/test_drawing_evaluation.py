import tf
import tf.transformations as tr
import rospy
import numpy as np

from arc_utilities.tf2wrapper import TF2Wrapper
from mmint_camera_utils.point_cloud_parsers import RealSensePointCloudParser



if __name__ == '__main__':
    # Parameters to tune
    tag_names = ['tag_5', 'tag_6', 'tag_7']
    camera_indx = 1
    board_x_size = 0.56
    board_y_size = 0.86
    tag_size = 0.09


    tf_listener = TF2Wrapper()
    rsp = RealSensePointCloudParser(camera_indx=camera_indx)
    camera_info_depth = rsp.get_camera_info_depth()
    camera_info_color = rsp.get_camera_info_color()
    camera_frame = 'camera_1_link'
    camera_depth_frame = 'camera_1_depth_opticla_frame' # TODO: replace with the one from camera_info
    tag_frames = ['{}_{}'.format(tag_name, camera_indx) for tag_name in tag_names]

    color_img_1 = rsp.get_image_color()
    depth_img_1 = rsp.get_image_depth()

    import pdb; pdb.set_trace()
    w_X_cf = np.eye(4) # TODO: Replace with the true camera-to-world calibrated tf
    cf_X_tags = [tf_listener.get_transform(parent=camera_frame, child=tag_frame) for tag_frame in tag_frames]
    w_X_tags = [w_X_cf @ cf_X_tag for cf_X_tag in cf_X_tags]

    # DO NOT TRUST THE ORIENTATION OF THE TAG SINCE IT CAN BE VERY NOISY