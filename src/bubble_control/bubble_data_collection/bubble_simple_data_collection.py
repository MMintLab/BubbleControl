import abc

import numpy as np
import rospy
from collections import OrderedDict
import gym
import copy
import time

from bubble_utils.bubble_data_collection.bubble_data_collection_base import BubbleDataCollectionBase


class BubbleSimpleDataCollection(BubbleDataCollectionBase):
    """
    Class designed to collect bubble data with the robot still. It is used to evaluate the bubble sensor noise.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def collect_data(self, num_data):
        print('Recording started. Moving the robot to the grasp pose')
        self.med.set_grasp_pose()
        _ = input('Press enter to open the gripper and calibrate the bubbles')
        self.med.open_gripper()
        print('Strating to collect data. Total data to be collected: {}'.format(num_data))
        out = super().collect_data(num_data)
        return out

    def _get_legend_column_names(self):
        column_names = ['Scene', 'FC', 'Time']
        return column_names

    def _get_legend_lines(self, data_params):
        legend_lines = []
        fc_i = data_params['fc']
        time_i = data_params['time']
        scene_i = self.scene_name
        line_i = [scene_i, fc_i, time_i]
        legend_lines.append(line_i)
        return legend_lines

    def _collect_data_sample(self, params=None):
        """
        Adjust the robot so the object has a constant pose (target_pose) in the reference ref_frame
        returns:
            data_params: <dict> containing the parameters and information of the collected data
        """
        data_params = {}
        # Sample the fcs:
        fc = self.get_new_filecode()
        time_i = time.time()
        self._record(fc=fc)
        data_params['fc'] = fc
        data_params['time'] = time_i
        return data_params


