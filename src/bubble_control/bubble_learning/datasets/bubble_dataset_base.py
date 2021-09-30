import numpy as np
import sys
import os
import pandas as pd
from torch.utils.data import Dataset
import abc
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from PIL import Image

from bubble_control.bubble_learning.datasets.dataset_base import DatasetBase
from mmint_camera_utils.point_cloud_utils import load_pointcloud


class BubbleDatasetBase(DatasetBase):

    def __init__(self, *args, **kwargs):
        self.wrench_columns = ['wrench.force.x', 'wrench.force.y', 'wrench.force.z', 'wrench.torque.x',
                               'wrench.torque.y', 'wrench.torque.z']
        self.tf_column_names = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        super().__init__(*args, **kwargs)

    def _get_filecodes(self):
        """
        Return a list containing the data filecodes.
        Overwrite the function in case the data needs to be filtered.
        By default we load all the filecodes we find in the datalegend
        :return:
        """
        return np.arange(len(self.dl))

    def _load_wrench(self, fc, scene_name):
        # return the saved wrench as DataFrame
        wrench_dir = os.path.join(self.data_path, scene_name, 'wrenches')
        wrench_file_path = os.path.join(wrench_dir, '{}_wrench_{:06}.csv'.format(scene_name, fc))
        wrench_df = pd.read_csv(wrench_file_path)
        return wrench_df

    def _load_tfs(self, fc, scene_name):
        # return the saved tfs as DataFrame
        tfs_dir = os.path.join(self.data_path, scene_name, 'tfs')
        tfs_file_path = os.path.join(tfs_dir, 'recorded_tfs_{:06}.csv'.format(fc))
        tfs_df = pd.read_csv(tfs_file_path)
        return tfs_df

    def _load_img(self, file_dir, file_name, extension=None):
        # Load an image as a np.ndarray. The image can be saved as .png or .npy
        if extension is None:
            # find what is the file extension
            all_files_in_file_dir = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
            our_files_in_file_dir = [f for f in all_files_in_file_dir if file_name in f]
            our_file = our_files_in_file_dir[0]
        else:
            our_file = '{}.{}'.format(file_name, extension)
        path_to_file = os.path.join(file_dir, our_file)
        # read files on dir
        if '.png' in our_file:
            img = Image.open(path_to_file)  # TODO: Test this
            img_array = np.array(img)
        elif '.npy' in our_file:
            img_array = np.load(path_to_file)
        else:
            img_array = None
        return img_array

    def _load_depth_img(self, fc, scene_name, camera_name):
        # depth can be saved as .png or .npy
        depth_dir = os.path.join(self.data_path, scene_name, 'pico_flexx_{}'.format(camera_name), 'depth_data')
        file_name = '{}_depth_{:06}'.format(scene_name, fc)
        depth_array = self._load_img(depth_dir, file_name)
        depth_array_meters = depth_array/10e9
        return depth_array_meters

    def _load_color_img(self, fc, scene_name, camera_name):
        color_dir = os.path.join(self.data_path, scene_name, 'pico_flexx_{}'.format(camera_name), 'color_data')
        file_name = '{}_color_{:06}'.format(scene_name, fc)
        color_array = self._load_img(color_dir, file_name, extension='png')
        return color_array

    def _get_depth_imprint(self, undef_fc, def_fc, scene_name, camera_name):
        # compare the depth image at undef_fc with the def_fc to obtain the imprint (i.e. deformation from the default state)
        undef_depth_img = self._load_depth_img(undef_fc, scene_name, camera_name)
        def_depth_img = self._load_depth_img(undef_fc, scene_name, camera_name)
        imprint = def_depth_img - undef_depth_img
        processed_imprint = self._process_bubble_img(imprint)
        return processed_imprint

    def _get_wrench(self, fc, scene_name, frame_id=None):
        raw_wrench_data = self._load_wrench(fc, scene_name)
        # return the wrench as a numpy array
        wrench_columns = self.wrench_columns
        frame_ids = raw_wrench_data['header.frame_id'].values
        if frame_id is None:
            # load all the data
            wrench = raw_wrench_data[wrench_columns].values
        elif frame_id in frame_ids:
            # return only the wrench for the given frame id
            wrench = raw_wrench_data[raw_wrench_data['header.frame_id'] == frame_id][wrench_columns].values
        else:
            # frame not found
            print('No frame named {} found. Available frames: {}'.format(frame_id, frame_ids))
            wrench = None
        return wrench

    def _get_tfs(self, fc, scene_name, frame_id=None):
        # frame_id is currently the name of the child frame
        raw_tfs = self._load_tfs(fc, scene_name)
        tf_column_names = self.tf_column_names
        frame_ids = raw_tfs['child_frame'].values
        if frame_id is None:
            # load tf all data
            tfs = raw_tfs[tf_column_names].values
        elif frame_id in frame_ids:
            # return only the tf for the given child frame
            tfs = raw_tfs[raw_tfs['child_frame'] == frame_id][tf_column_names].values
        else:
            # frame not found
            print('No frame named {} found. Available frames: {}'.format(frame_id, frame_ids))
            tfs = None
        return tfs

    def _load_pc(self, fc, scene_name, camera_name):
        pc_dir = os.path.join(self.data_path, scene_name, 'pico_flexx_{}'.format(camera_name), 'point_cloud_data')
        fc_file_path = os.path.join(pc_dir, '{}_pc_{:06}.ply'.format(scene_name, fc))
        # load the .ply as a point cloud
        pc_array = load_pointcloud(fc_file_path, as_array=True)
        return pc_array

    # TODO; Consider adding other load methods like camera model

    def _process_bubble_img(self, bubble_img):
        # remove the noisy areas of the images:
        bubble_img_out = bubble_img[20:160, 25:200]
        return bubble_img_out
