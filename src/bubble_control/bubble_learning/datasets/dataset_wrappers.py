import numpy as np
import tf.transformations as tr

from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase
from bubble_control.bubble_learning.aux.img_trs.block_downsampling_tr import BlockDownSamplingTr
from mmint_camera_utils.aux.wrapping_utils import AttributeWrapper, ClassWrapper, DecoratorWrapper


def imprint_downsampled_dataset(cls):
    class Wrapper(AttributeWrapper):
        def __init__(self, *args, downsample_factor_x=5, downsample_factor_y=5, downsample_reduction='mean', **kwargs):
            self.downsample_factor_x = downsample_factor_x
            self.downsample_factor_y = downsample_factor_y
            self.downsample_reduction = downsample_reduction
            self.block_mean_downsampling_tr = BlockDownSamplingTr(factor_x=downsample_factor_x,
                                                                  factor_y=downsample_factor_y,
                                                                  reduction=self.downsample_reduction)  # downsample all imprint values

            # add the block_mean_downsampling_tr to the tr list
            if 'transformation' in kwargs:
                if type(kwargs['transformation']) in (list, tuple):
                    kwargs['transformation'] = list(kwargs['transformation']) + [self.block_mean_downsampling_tr]
                else:
                    print('')
                    raise AttributeError('Not supportes trasformations: {} type {}'.format(kwargs['transformation'],
                                                                                           type(kwargs[
                                                                                                    'transformation'])))
            else:
                kwargs['transformation'] = [self.block_mean_downsampling_tr]
            super().__init__(cls.__init__(*args, **kwargs))

        @classmethod
        def get_name(self):
            return '{}_downsampled'.format(self.wrapped_object.get_name())

    return Wrapper


class CombinedDatasetWrapper(ClassWrapper):
    @classmethod
    def get_name(self):
        return '{}_combined'.format(self.wrapped_object.get_name())

    def _get_filecodes(self):
        # duplicate the filecodes:
        fcs = np.arange(2 * len(self.wrapped_object._get_filecodes()))
        return fcs

    def _get_sample(self, indx):
        # fc: index of the line in the datalegend (self.dl) of the sample
        true_indx = indx // 2
        dl_line = self.dl.iloc[true_indx]
        sample = self.wrapped_object._get_sample(true_indx)
        if indx % 2 == 0:
            # sample is the initial
            sample['imprint'] = sample['init_imprint']
        else:
            sample['imprint'] = sample['final_imprint']
        return sample


combined_dataset = DecoratorWrapper(CombinedDatasetWrapper)