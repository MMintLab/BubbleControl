import numpy as np

from bubble_utils.bubble_tools.bubble_img_tools import process_bubble_img





def format_observation_sample(obs_sample):
    formatted_obs_sample = {}
    # obs sample should have:
    #           'init_imprint',
    #           'init_wrench',
    #           'init_pos',
    #           'init_quat',
    #           'final_imprint',
    #           'final_wrench',
    #           'final_pos',
    #           'final_quat',
    #           'action',
    #           'undef_depth_r',
    #           'undef_depth_l',
    #           'camera_info_r',
    #           'camera_info_l',
    #           'all_tfs'
    # input input expected keys:
    #           'bubble_camera_info_color_right',
    #           'bubble_camera_info_depth_right',
    #           'bubble_color_img_right',
    #           'bubble_depth_img_right',
    #           'bubble_point_cloud_right',
    #           'bubble_camera_info_color_left',
    #           'bubble_camera_info_depth_left',
    #           'bubble_color_img_left',
    #           'bubble_depth_img_left',
    #           'bubble_point_cloud_left',
    #           'wrench',
    #           'tfs',
    #           'bubble_color_img_right_reference',
    #           'bubble_depth_img_right_reference',
    #           'bubble_point_cloud_right_reference',
    #           'bubble_color_img_left_reference',
    #           'bubble_depth_img_left_reference',
    #           'bubble_point_cloud_left_reference'
    # remap keys ---
    key_map = {
        'tfs': 'all_tfs',
        'bubble_camera_info_depth_left': 'camera_info_l',
        'bubble_camera_info_depth_right': 'camera_info_r',
        'bubble_depth_img_right_reference': 'undef_depth_r',
        'bubble_depth_img_left_reference': 'undef_depth_l',
    }
    for k_old, k_new in key_map.items():
        formatted_obs_sample[k_new] = obs_sample[k_old]
    # add imprints: -------
    init_imprint_r = obs_sample['bubble_depth_img_right_reference'] - obs_sample['bubble_depth_img_right']
    init_imprint_l = obs_sample['bubble_depth_img_left_reference'] - obs_sample['bubble_depth_img_left']
    formatted_obs_sample['init_imprint'] = process_bubble_img(np.stack([init_imprint_r, init_imprint_l], axis=0))[..., 0]

    # apply the key_map
    return formatted_obs_sample