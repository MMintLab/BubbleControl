import abc
import torch
import numpy as np
import copy
import tf.transformations as tr
from pytorch_mppi import mppi
import pytorch3d.transforms as batched_trs

from bubble_control.bubble_model_control.controllers.bubble_controller_base import BubbleModelController
from bubble_control.bubble_model_control.aux.bubble_model_control_utils import batched_tensor_sample, get_transformation_matrix, tr_frame, convert_all_tfs_to_tensors
from bubble_control.bubble_model_control.controllers.bubble_model_mppi_controler import BubbleModelMPPIController


class DrawingMPPIController(BubbleModelMPPIController):
    """
    Batched controller with a batched pose estimation
    """
    def _get_state_keys(self):
        state_keys = ['init_imprint', 'init_wrench', 'init_pos', 'init_quat', 'object_model'] # TODO: Verify that we can get init_quat and init_pos from sample
        return state_keys

    def _get_model_output_keys(self):
        output_keys = ['init_imprint', 'init_wrench']
        return output_keys

    def _get_next_state_map(self):
        next_state_map = {
            'init_imprint': 'next_imprint',
            'init_wrench': 'final_wrench',

        }
        return next_state_map


class DrawingMPPISimpleController(BubbleModelMPPIController):
    """
    Batched controller with a batched pose estimation.
    Only considers imprints.
    """
    def _get_state_keys(self):
        state_keys = ['init_imprint']
        return state_keys

    def _get_model_output_keys(self):
        output_keys = ['init_imprint']
        return output_keys

    def _get_next_state_map(self):
        next_state_map = {
            'init_imprint': 'next_imprint'
        }
        return next_state_map

