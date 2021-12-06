import numpy as np
import os
import sys
import torch

from bubble_control.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from bubble_control.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from bubble_control.bubble_learning.aux.img_trs.block_upsampling_tr import BlockUpSamplingTr
from bubble_control.bubble_learning.models.bubble_dynamics_pretrained_ae_model import BubbleDynamicsPretrainedAEModel
from bubble_control.aux.load_confs import load_bubble_reconstruction_params
from bubble_control.bubble_pose_estimation.bubble_pc_reconstruction import BubblePCReconsturctorDepth
from bubble_utils.bubble_tools.bubble_img_tools import unprocess_bubble_img



class BubblePCReconsturctorOfflineDepth(BubblePCReconsturctorDepth):
    def __init__(self, *args, **kwargs):
        self.depth_r = None
        self.depth_l = None
        super().__init__(*args, **kwargs)

    def _get_depth_imgs(self):
        depth_r = self.depth_r
        depth_l = self.depth_l
        return depth_r, depth_l


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

    predicted_imprint = sample_up['next_imprint']
    imprint_pred_r, imprint_pred_l = predicted_imprint

    #  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   Object Pose Estimation   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # TODO: Consider extending BubblePCReconstructorDepth with a modified get_imprint methods

    # estimate the contact imprint for each bubble (right and left)
    object_name = 'marker'
    reconstruction_params = load_bubble_reconstruction_params()
    object_params = reconstruction_params[object_name]
    imprint_threshold = object_params['imprint_th']['depth']
    icp_threshold = object_params['icp_th']

    reconstructor = BubblePCReconsturctorOfflineDepth(threshold=imprint_threshold, object_name=object_name, estimation_type='icp2d')
    # obtain camera parameters
    camera_info_r = sample['camera_info_r']
    camera_info_l = sample['camera_info_l']
    all_tfs = sample['all_tfs']

    # obtain reference (undeformed) depths
    ref_depth_img_r = sample['undef_depth_l']
    ref_depth_img_l = sample['undef_depth_l']

    # unprocess the imprints (add padding to move them back to the original shape)
    imprint_pred_r = unprocess_bubble_img(imprint_pred_r)
    imprint_pred_l = unprocess_bubble_img(imprint_pred_l)

    deformed_depth_r = ref_depth_img_r + imprint_pred_r
    deformed_depth_l = ref_depth_img_l + imprint_pred_l

    # THIS hacks the ways to obtain data for the reconstructor
    reconstructor.references['left'] = deformed_depth_l
    reconstructor.references['right'] = deformed_depth_r
    reconstructor.depth_r = deformed_depth_r
    reconstructor.depth_l = deformed_depth_l
    reconstructor.camera_info['right'] = camera_info_r
    reconstructor.camera_info['left'] = camera_info_l
    #           add also the tfs to the parsers's tf buffer
    # estimate pose
    estimated_pose = reconstructor.estimate_pose(icp_threshold)