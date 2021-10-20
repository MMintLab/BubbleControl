import numpy as np

from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase


class BubbleDrawingDataset(BubbleDatasetBase):

    def __init__(self, *args, wrench_frame=None, tf_frame='grasp_frame', **kwargs):
        self.wrench_frame = wrench_frame
        self.tf_frame = tf_frame
        super().__init__(*args, **kwargs)

    @property
    def name(self):
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
        final_wrench = self._get_wrench(fc=init_fc, scene_name=scene_name, frame_id=self.wrench_frame)

        # TODO: Add frames on sample
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
        action_column_names = ['GraspForce', 'grasp_width', 'direction', 'length']
        dl_line = self.dl.iloc[fc]
        action_i = dl_line[action_column_names].values.astype(np.float64)
        return action_i

    def _compute_delta_sample(self, sample):
        # TODO: improve this computation
        input_keys = ['imprint', 'wrench', 'pos', 'quat']
        time_keys = ['init', 'final']
        for in_key in input_keys:
            sample['delta_{}'.format(in_key)] = sample['final_{}'.format(in_key)] - sample['init_{}'.format(in_key)]
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

# DEBUG:

if __name__ == '__main__':
    data_name = '/home/mmint/Desktop/drawing_data'
    dataset = BubbleDrawingDataset(data_name=data_name)
    print('Dataset Name: ', dataset.name)
    print('Dataset Length:', len(dataset))
    sample_0 = dataset[0]
    print('Sample 0:', sample_0)
