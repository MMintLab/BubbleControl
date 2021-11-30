import numpy as np
import torch
from scipy import ndimage

class BlockMeanDownSamplingTr(object):

    def __init__(self, factor, keys_to_tr):
        self.factor = factor
        self.keys_to_tr = keys_to_tr

    def __call__(self, sample):
        if self.keys_to_tr is None:
            # transform all that has quat in the key
            for k, v in sample.items():
                if 'imprint' in k:
                    sample['{}_undownsampled'.format(k)] = v  # store the unsampled one
                    sample[k] = self._tr(v)
        else:
            for key in self.keys_to_tr:
                if key in sample:
                    sample['{}_undownsampled'.format(key)] = v  # store the unsample one
                    sample[key] = self._tr(sample[key])
        return sample

    def inverse(self, sample):
        # apply the inverse transformation
        if self.keys_to_tr is None:
            # trasform all that has quat in the key
            for k, v in sample.items():
                if 'imprint' in k:
                    sample[k] = sample['{}_undownsampled'.format(k)] # restore the original
        else:
            for key in self.keys_to_tr:
                if key in sample:
                    sample[key] = sample['{}_undownsampled'.format(key)] # restore the original
        return sample

    def _tr(self, x):
        # downsample the image using block mean
        import pdb; pdb.set_trace()
        size_x, size_y = None, None
        X, Y = np.ogrid[0:size_x, 0:size_y]
        regions = size_y/self.factor * (X/self.factor) + Y/self.factor
        x_down = ndimage.mean(x, labels=regions, index=np.arange(regions.max() + 1))
        import pdb; pdb.set_trace()
        return x_down