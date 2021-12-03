import numpy as np
from scipy import interpolate
import copy
import abc


class BlockUpSamplingTr(abc.ABC):
    def __init__(self, factor_x, factor_y, method='repeat', keys_to_tr=None):
        super().__init__()
        self.factor_x = factor_x
        self.factor_y = factor_y
        self.method = method
        self.keys_to_tr = keys_to_tr

    def __call__(self, sample):
        if self.keys_to_tr is None:
            # transform all that has quat in the key
            old_keys = copy.deepcopy(list(sample.keys()))
            for k in old_keys:
                v = sample[k]
                if 'imprint' in k:
                    sample['{}_upsampled'.format(k)] = v  # store the unsampled one
                    sample[k] = self._tr(v)
        else:
            for key in self.keys_to_tr:
                if key in sample:
                    v = sample[key]
                    sample['{}_upsampled'.format(key)] = v  # store the unsample one
                    sample[key] = self._tr(sample[key])
        return sample

    def inverse(self, sample):
        # apply the inverse transformation
        if self.keys_to_tr is None:
            # trasform all that has quat in the key
            for k, v in sample.items():
                if 'imprint' in k:
                    sample[k] = sample['{}_upsampled'.format(k)] # restore the original
        else:
            for key in self.keys_to_tr:
                if key in sample:
                    sample[key] = sample['{}_upsampled'.format(key)] # restore the original
        return sample

    def _tr(self, x):
        # ---- repeat upsampling ----
        if self.method == 'repeat':
            x_upsampled = x.repeat(self.factor_x, axis=-2).repeat(self.factor_y, axis=-1)
        elif self.method == 'interpolation':
            # ---- interpolation upsampling -- (TODO)
            # TODO: Add
            raise NotImplemented('method interpolation not available yet')
        else:
            raise NotImplemented('method {} not available yet. Available methods: {}'.format(self.metod, ['repeat']))
        return x_upsampled

