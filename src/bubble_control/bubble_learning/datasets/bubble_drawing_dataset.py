import numpy as np
import tf.transformations as tr

from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase
from bubble_control.bubble_learning.aux.img_trs import BlockDownSamplingTr

class BubbleDrawingDataset(BubbleDatasetBase):

    def __init__(self, *args, wrench_frame=None, tf_frame='grasp_frame', **kwargs):
        self.wrench_frame = wrench_frame
        self.tf_frame = tf_frame
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(self):
        return 'bubble_drawing_dataset'

    def _get_sample(self, fc):
        # fc: index of the line in the datalegend (self.dl) of the sample
        dl_line = self.dl.iloc[fc]
        scene_name = dl_line['Scene']
        undef_fc = dl_line['UndeformedFC']
        init_fc = dl_line['InitialStateFC']
        final_fc = dl_line['FinalStateFC']
        # Load initial state:
        init_imprint_r = self._get_depth_imprint(undef_fc=undef_fc, def_fc=init_fc, scene_name=scene_name, camera_name='right')
        init_imprint_l = self._get_depth_imprint(undef_fc=undef_fc, def_fc=init_fc, scene_name=scene_name, camera_name='left')
        init_imprint = np.stack([init_imprint_r, init_imprint_l], axis=0)
        init_wrench = self._get_wrench(fc=init_fc, scene_name=scene_name, frame_id=self.wrench_frame)
        # Final State
        final_imprint_r = self._get_depth_imprint(undef_fc=undef_fc, def_fc=final_fc, scene_name=scene_name, camera_name='right')
        final_imprint_l = self._get_depth_imprint(undef_fc=undef_fc, def_fc=final_fc, scene_name=scene_name, camera_name='left')
        final_imprint = np.stack([final_imprint_r, final_imprint_l], axis=0)
        final_wrench = self._get_wrench(fc=final_fc, scene_name=scene_name, frame_id=self.wrench_frame)

        init_tf = self._get_tfs(init_fc, scene_name=scene_name, frame_id=self.tf_frame)
        final_tf = self._get_tfs(final_fc, scene_name=scene_name, frame_id=self.tf_frame)
        init_pos = init_tf[..., :3]
        init_quat = init_tf[..., 3:]
        final_pos = final_tf[..., :3]
        final_quat = final_tf[..., 3:]

        # Action:
        action_fc = fc
        action = self._get_action(action_fc)

        sample_simple = {
            'init_imprint': init_imprint,
            'init_wrench': init_wrench,
            'init_pos': init_pos,
            'init_quat': init_quat,
            'final_imprint': final_imprint,
            'final_wrench': final_wrench,
            'final_pos': final_pos,
            'final_quat': final_quat,
            'action': action,
        }
        sample = self._reshape_sample(sample_simple)
        sample = self._compute_delta_sample(sample) # Add delta values to sample

        return sample

    def _get_action(self, fc):
        # TODO: Load from file instead of the logged values in the dl
        dl_line = self.dl.iloc[fc]
        # action_column_names = ['GraspForce', 'grasp_width', 'direction', 'length']
        # action_i = dl_line[action_column_names].values.astype(np.float64)
        direction = dl_line['direction']
        length = dl_line['length']
        action_i = length * np.array([np.cos(direction), np.sin(direction)])
        return action_i

    def _compute_delta_sample(self, sample):
        # TODO: improve this computation
        input_keys = ['imprint', 'wrench', 'pos']
        time_keys = ['init', 'final']
        for in_key in input_keys:
            sample['delta_{}'.format(in_key)] = sample['final_{}'.format(in_key)] - sample['init_{}'.format(in_key)]
        # for quaternion, compute the delta quaternion q_delta @ q_init = q_final <=> q_delta
        sample['delta_quat'] = tr.quaternion_multiply(sample['final_quat'], tr.quaternion_inverse(sample['init_quat']))
        return sample

    def _reshape_sample(self, sample):
        input_keys = ['wrench', 'pos', 'quat']
        time_keys = ['init', 'final']
        # reshape the imprint
        for time_key in time_keys:
            imprint_x = sample['{}_imprint'.format(time_key)]
            sample['{}_imprint'.format(time_key)] = imprint_x.transpose((0,3,1,2)).reshape(-1, *imprint_x.shape[1:3])
            for in_key in input_keys:
                sample['{}_{}'.format(time_key, in_key)] = sample['{}_{}'.format(time_key, in_key)].flatten()
        sample['action'] = sample['action'].flatten()
        return sample


class BubbleDrawingDownsampledDataset(BubbleDrawingDataset):
    def __init__(self, *args, downsample_factor_x=5, downsample_factor_y=5, downsample_reduction='mean', **kwargs):
        self.downsample_factor_x = downsample_factor_x
        self.downsample_factor_y = downsample_factor_y
        self.downsample_reduction = downsample_reduction
        self.block_mean_downsampling_tr = BlockDownSamplingTr(factor_x=downsample_factor_x, factor_y=downsample_factor_y, reduction=self.downsample_reduction) #downsample all imprint values
        # add the block_mean_downsampling_tr to the tr list
        if 'transformation' in kwargs:
            if type(kwargs['transformation']) in (list, tuple):
                kwargs['transformation'] = list(kwargs['transformation']) + [self.block_mean_downsampling_tr]
            else:
                print('')
                raise AttributeError('Not supportes trasformations: {} type {}'.format(kwargs['transformation'], type(kwargs['transformation'])))
        else:
            kwargs['transformation'] = [self.block_mean_downsampling_tr]
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(self):
        return 'bubble_drawing_downsampled_dataset'

    # @property
    # def name(self):
    #     """
    #     Returns an unique identifier of the dataset
    #     :return:
    #     """
    #     # Override this so every reduction has its name
    #     return '{}_fx_fy_self.get_name()


# DEBUG:

if __name__ == '__main__':
    data_name = '/home/mmint/Desktop/drawing_data_cartesian'
    dataset = BubbleDrawingDataset(data_name=data_name, wrench_frame='med_base', tf_frame='grasp_frame')
    print('Dataset Name: ', dataset.name)
    print('Dataset Length:', len(dataset))
    sample_0 = dataset[0]
    print('Sample 0:', sample_0)
