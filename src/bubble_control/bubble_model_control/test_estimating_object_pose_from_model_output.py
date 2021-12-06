import numpy as np
import os
import sys
import torch

from bubble_control.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from bubble_control.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from bubble_control.bubble_learning.aux.img_trs.block_upsampling_tr import BlockUpSamplingTr
from bubble_control.bubble_learning.models.bubble_dynamics_pretrained_ae_model import BubbleDynamicsPretrainedAEModel



if __name__ == '__main__':

    data_name = '/home/mik/Desktop/drawing_data_cartesian'
    dataset = BubbleDrawingDataset(data_name=data_name, wrench_frame='med_base', tf_frame='grasp_frame')
    Model = BubbleDynamicsPretrainedAEModel

    load_version = 11

    block_downsample_tr = BlockDownSamplingTr(factor_x=7, factor_y=7, reduction='max', keys_to_tr=['init_imprint'])
    block_upsample_tr = BlockUpSamplingTr(factor_x=7, factor_y=7, method='bilinear')

    # load model:
    model_name = Model.get_name()
    version_chkp_path = os.path.join(data_name, 'tb_logs', '{}'.format(model_name),
                                     'version_{}'.format(load_version), 'checkpoints')
    checkpoints_fs = [f for f in os.listdir(version_chkp_path) if
                      os.path.isfile(os.path.join(version_chkp_path, f))]
    checkpoint_path = os.path.join(version_chkp_path, checkpoints_fs[0])

    model = Model.load_from_checkpoint(checkpoint_path)
    import pdb; pdb.set_trace()

    # load one sample:
    sample = dataset[0]

    # Downsample
    sample['init_imprint'] = sample['init_imprint']
    sample_down = block_downsample_tr(sample)

    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Query model   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    imprint_t = torch.tensor(sample_down['init_imprint']).unsqueeze(0).to(dtype=torch.float)
    action_t = torch.tensor(sample_down['action']).unsqueeze(0).to(dtype=torch.float)

    # predict next imprint
    next_imprint = model(imprint_t, action_t).squeeze()

    next_imprint = next_imprint.cpu().detach().numpy()
    # Upsample ouptut
    sample_out = {
        'next_imprint':next_imprint,
    }
    sample_up = block_upsample_tr(sample_out)

    predicted_imprint_state = sample_up['next_imprint']

    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Object Pose Estimation   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

