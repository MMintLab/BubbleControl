import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from bubble_control.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from bubble_control.bubble_learning.models.bubble_dynamics_model import BubbleDynamicsModel
from bubble_control.bubble_model_control.aux.bubble_dynamics_fixed_model import BubbleDynamicsFixedModel
from bubble_control.bubble_learning.models.bubble_linear_dynamics_model import BubbleLinearDynamicsModel
from bubble_control.bubble_learning.models.object_pose_dynamics_model import ObjectPoseDynamicsModel

from bubble_control.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from bubble_control.bubble_learning.datasets.fixing_datasets.fix_object_pose_encoding_processed_data import EncodeObjectPoseAsAxisAngleTr
from bubble_control.bubble_learning.aux.orientation_trs import QuaternionToAxis
from bubble_control.bubble_learning.aux.remove_nontensor_elements_tr import RemoveNonTensorElementsTr
from bubble_control.bubble_learning.aux.load_model import load_model_version
from bubble_utils.bubble_datasets.dataset_transformed import transform_dataset


if __name__ == '__main__':

    data_path = '/home/mmint/Desktop/model_debugging_drawing'

    trs = [QuaternionToAxis(), EncodeObjectPoseAsAxisAngleTr()]

    dataset = BubbleDrawingDataset(
            data_name=data_path,
            downsample_factor_x=7,
            downsample_factor_y=7,
            downsample_reduction='mean',
            wrench_frame='med_base',
            dtype=torch.float32,
            transformation=trs,
            tf_frame='grasp_frame',
            contribute_mode=True,
            clean_if_error=True,
        )
    remove_nontensor_elements_tr = RemoveNonTensorElementsTr()
    trs = (remove_nontensor_elements_tr,)
    dataset = transform_dataset(dataset, transforms=trs)

    model_data_path = '/home/mmint/Desktop/drawing_models'
    model_version = 0
    Model = BubbleDynamicsModel
    model = load_model_version(Model, model_data_path, load_version=model_version)
    model.eval()

    dl = DataLoader(dataset, batch_size=len(dataset), num_workers=8, drop_last=True)

    outs = []
    ins = []
    for b_i, batch_i in enumerate(dl):
        model_input_i = model.get_model_input(batch_i)
        ground_truth_i = model.get_model_output(batch_i)
        action_i = batch_i['action']
        out_i = model(*model_input_i, action_i)
        outs.append(out_i)
        ins.append(model_input_i)
        # Visualize the imprints:
        num_to_visualize = 10
        fig, axes = plt.subplots(nrows=num_to_visualize, ncols=6)
        for v_indx in range(num_to_visualize):
            # columns: init_imprint_l, next_imprint_gth_l, next_imprint_pred_l, init_imprint_r, next_imprint_gth_r, next_imprint_pred_r,
            axes[v_indx][0].imshow(model_input_i[0][v_indx, 1].detach().numpy(), cmap='jet')
            axes[v_indx][1].imshow(ground_truth_i[0][v_indx, 1].detach().numpy(), cmap='jet')
            axes[v_indx][2].imshow(out_i[0][v_indx, 1].detach().numpy(), cmap='jet')
            axes[v_indx][3].imshow(model_input_i[0][v_indx, 0].detach().numpy(), cmap='jet')
            axes[v_indx][4].imshow(ground_truth_i[0][v_indx, 0].detach().numpy(), cmap='jet')
            axes[v_indx][5].imshow(out_i[0][v_indx, 0].detach().numpy(), cmap='jet')
        column_names = ['init_imprint_l', 'next_imprint_gth_l', 'next_imprint_pred_l', 'init_imprint_r', 'next_imprint_gth_r', 'next_imprint_pred_r']
        for c_i, cn in enumerate(column_names):
            axes[0][c_i].set_title(cn)
        plt.show()
    # ------------------------------------------------------------------------------------------------------------------
    # CONCLUSION: -
    # --> The imprints look fairly close to the predicted ones, and therefore the bug is somewhere on the data tranformation. Maybe wrench scale, position, or other.



