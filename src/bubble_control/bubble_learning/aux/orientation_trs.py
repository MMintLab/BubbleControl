import torch
import numpy as np
import tf.transformations as tr


class QuaternionToAxis(object):

    def __init__(self, keys_to_tr=None):
        self.keys_to_tr = keys_to_tr

    def __call__(self, sample):
        if self.keys_to_tr is None:
            # transform all that has quat in the key
            for k, v in sample.items():
                if 'quat' in k:
                    sample[k] = self._tr(v)
        else:
            for key in self.keys_to_tr:
                if key in sample:
                    sample[key] = self._tr(sample[key])
        return sample

    def inverse(self, sample):
        # apply the inverse transformation
        if self.keys_to_tr is None:
            # trasform all that has quat in the key
            for k, v in sample.items():
                if 'quat' in k:
                    sample[k] = self._tr_inv(v)
        else:
            for key in self.keys_to_tr:
                if key in sample:
                    sample[key] = self._tr_inv(sample[key])
        return sample

    def _tr(self, x):
        # transform a quaternion encoded rotation to an axis one with 3 values representing the axis of rotation where the modulus is the angle magnitude
        # q = [qx, qy, qz, qw] where qw = cos(theta/2); qx = a1*sin(theta/2),...
        qw = x[..., -1]
        theta = 2 * np.arccos(qw)
        axis = x[..., :3] / np.sin(theta/2) # should be a unit vector
        x_tr = theta * axis
        return x_tr

    def _tr_inv(self, x_tr):
        theta = np.linalg.norm(x_tr, axis=-1)
        axis = x_tr/theta
        qw = np.cos(theta/2)
        qxyz = np.sin(theta/2)*axis
        x = np.append(qxyz, 3, qw, axis=-1)
        return x

