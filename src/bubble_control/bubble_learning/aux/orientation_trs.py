import torch
import numpy as np
import tf.transformations as tr
import pytorch3d.transforms as batched_trs

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
        if len(x.shape) == 2:
            theta = np.expand_dims(theta, axis=1)        
        axis = x[..., :3] / np.sin(theta/2) # should be a unit vector
        x_tr = theta * axis
        return x_tr

    def _tr_inv(self, x_tr):
        theta = np.linalg.norm(x_tr, axis=-1)
        if len(x_tr.shape) == 2:
            theta = np.expand_dims(theta, axis=1)
        axis = x_tr/theta
        qw = np.cos(theta/2)
        qxyz = np.sin(theta/2)*axis
        x = np.append(qxyz, qw, axis=-1)
        return x

class EulerToAxis(object):
    def __init__(self):
        self.quat_to_axis = QuaternionToAxis()

    def euler_sxyz_to_axis_angle(self, euler_sxyz):
        # transform an euler encoded rotation to an axis one with 3 values representing the axis of rotation where the modulus is the angle magnitude
        if euler_sxyz.type == np.ndarray:
            euler_sxyz = torch.from_numpy(euler_sxyz, requires_grad=False)
        euler_reordered = torch.index_select(euler_sxyz, dim=-1, index=torch.LongTensor([2, 1, 0]))
        matrix = batched_trs.euler_angles_to_matrix(euler_reordered, 'ZYX')
        quaternion_wxyz = batched_trs.matrix_to_quaternion(matrix)
        quaternion = torch.index_select(quaternion_wxyz, dim=-1, index=torch.LongTensor([1, 2, 3, 0]))
        axis_angle = torch.from_numpy(self.quat_to_axis._tr(quaternion.detach().numpy()))
        return axis_angle

    def axis_angle_to_euler_sxyz(self, axis_angle):
        matrix = batched_trs.axis_angle_to_matrix(axis_angle)
        euler = batched_trs.matrix_to_euler_angles(matrix, 'ZYX')
        euler_sxyz = torch.index_select(euler, dim=-1, index=torch.LongTensor([2,1,0]))
        return euler_sxyz