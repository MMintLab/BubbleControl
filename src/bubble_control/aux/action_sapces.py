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