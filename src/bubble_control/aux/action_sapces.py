import abc
import numpy as np
from collections import OrderedDict
import gym
import copy


class AxisBiasedDirectionSpace(gym.spaces.Space):
    """
    Saple space between [0,2pi) with bias towards the axis directions.
    On prob_axis, the sample will be along one of the cartesian axis directions, i.e. [0, pi/2, pi, 3pi/2]
    """
    def __init__(self, prob_axis, seed=None):
        """
        Args:
            prob_axis: probability of sampling a direction along the axis
            seed:
        """
        self.prob_axis = prob_axis
        super().__init__((), np.float32, seed)

    def sample(self):
        p_axis_direction = self.np_random.random() # probability of getting an axis motion
        if p_axis_direction < self.prob_axis:
            direction_i = 0.5 * np.pi * np.random.randint(0, 4) # axis direction (0, pi/2, pi/ 3pi/2)
        else:
            direction_i = np.random.uniform(0, 2 * np.pi)  # direction as [0, 2pi)
        return direction_i

    def contains(self, x):
        return 0 <= x <= 2*np.pi


class FinalPivotingPoseSpace(gym.spaces.Space):
    """
    Sample pivoting pose (with orientation as quaternion)
    """
    def __init__(self, med, current_pose, delta_y_limits, delta_z_limits, delta_roll_limits, seed=None):
        super().__init__((), np.float32, seed)
        self.current_pose = current_pose
        self.delta_y_limits = delta_y_limits
        self.delta_z_limits = delta_z_limits
        self.delta_roll_limits = delta_roll_limits
        self.med = med

    def sample(self):
        delta_y, delta_z = np.random.uniform(np.array([self.delta_y_limits[0], self.delta_z_limits[0]]), 
                                            np.array([self.delta_y_limits[1], self.delta_z_limits[0]]))
        movement_wf = delta_y * np.array([0,1,0]) + delta_z * np.array([0,0,1])
        delta_roll_wf = np.random.uniform(self.delta_roll_limits[0], self.delta_roll_limits[1])
        orientation = self.med._compute_rotation_along_axis_point_angle(pose=self.current_pose, 
                        angle=delta_roll_wf, point=self.current_pose[:3], axis=np.array([1,0,0]))[3:]
        final_pose_wf = np.concatenate([self.current_pose[:3]+movement_wf, orientation])
        return final_pose_wf

    #TODO: Add orientation limits
    def contains(self, position):
        lower_bound = self.current_pose[:3] + np.array([0, self.delta_y_limits[0], self.delta_z_limits[0]])
        upper_bound = self.current_pose[:3] + np.array([0, self.delta_y_limits[1], self.delta_z_limits[1]])
        return lower_bound  <= position <= upper_bound

class InitialPivotingPoseSpace(gym.spaces.Space):
    """
    Sample initial pose for pivoting (with orientation as euler)
    """
    def __init__(self, init_x_limits, init_y_limits, init_z_limits, roll_limits, seed=None):
        super().__init__((), np.float32, seed)
        self.init_x_limits = init_x_limits
        self.init_y_limits = init_y_limits
        self.init_z_limits = init_z_limits
        self.roll_limits = roll_limits

    def sample(self):
        initial_position = np.random.uniform(np.array([self.init_x_limits[0], self.init_y_limits[0], self.init_z_limits[0]]),
                                             np.array([self.init_x_limits[1], self.init_y_limits[1], self.init_z_limits[1]]))
        roll = np.random.uniform(self.roll_limits[0],self.roll_limits[1])
        initial_orientation = np.array([roll, 0, np.pi])
        initial_pose_wf = np.concatenate([initial_position, initial_orientation])                                            
        return initial_pose_wf

    def contains(self, pose):
        lower_bound = np.array([self.init_x_limits[0], self.init_y_limits[0], self.init_z_limits[0], self.roll_limits[0], 0, np.pi])
        upper_bound = np.array([self.init_x_limits[1], self.init_y_limits[1], self.init_z_limits[1], self.roll_limits[1], 0, np.pi])
        return lower_bound  <= pose <= upper_bound

class ConstantSpace(gym.spaces.Space):
    """
    Constant space. Only has one possible value. For convenience.
    """
    def __init__(self, value, seed=None):
        self.value = value
        super().__init__((), np.float32, seed)

    def sample(self):
        return self.value

    def contains(self, x):
        return x == self.value

    def __eq__(self, other):
        return (
                isinstance(other, ConstantSpace)
                and self.value == other.value
        )