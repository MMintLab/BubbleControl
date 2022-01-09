import abc
import torch
import numpy as np
import copy
from pytorch_mppi import mppi


class BubbleModelController(abc.ABC):

    def __init__(self, model, object_pose_estimator, cost_function):

        self.model = model
        self.object_pose_estimator = object_pose_estimator
        self.cost_function = cost_function
        self.controller = self._get_controller()

    def control(self, state_sample):
        action = self._query_controller(state_sample)
        return action

    @abc.abstractmethod
    def _get_controller(self):
        pass

    @abc.abstractmethod
    def _query_controller(self, state_sample):
        pass


class BubbleModelMPPIController(BubbleModelController):

    def __init__(self, *args, num_samples=100, horizon=3,**kwargs):
        self.num_samples = num_samples
        self.horizon = horizon
        self.original_state_shape = None
        self.noise_sigma = None
        self.state_size = None # To be filled with sample information or model information
        self.sample = None # Container to share sample across functions
        super().__init__(*args, **kwargs)
        self.state_size = np.prod(self.model._get_sizes()['imprint']) # TODO: Make more general

    def _init_params(self):
        # TODO: Make more general
        self.original_state_shape = self.model.input_sizes['init_imprint']
        self.state_size = np.prod(self.original_state_shape)
        self.noise_sigma = torch.tensor(np.diag([np.pi/2,.05, 0.05]), device=self.model.device, dtype=torch.float)

    def dynamics(self, state_t, action_t):
        """
        Compute the dynamics
        :param state_t: (K, state_size) tensor
        :param action_t: (K, action_size) tensor
        :return: next_state: (K, state_size) tensor
        """

        # TODO: Query model
        state = self._unpack_state_tensor(state_t)
        action = self._unpack_action_tensor(action_t)
        next_state = self.model(state, action)

        next_state_t = self._pack_state_to_tensor(next_state)

        return next_state_t

    def compute_cost(self, state_t, action_t):
        """
        Compute the dynamics
        :param state: (K, state_size) tensor
        :param action: (K, action_size) tensor
        :return: cost: (K, 1) tensor
        """
        # TODO: compute cost (use self.cost_function and object_pose estimator)
        costs_t = torch.zeros((len(state_t), 1))

        states = self._unpack_state_tensor(state_t)
        actions = self._unpack_action_tensor(action_t)

        estimated_poses = []
        for i, state_i in enumerate(states):
            state_sample_i = self._pack_state_to_sample(state_i, self.sample)
            estimated_pose_i = self.object_pose_estimator.estimate_pose(state_sample_i)
            estimated_poses.append(estimated_pose_i)
        estimated_poses = np.array(estimated_poses)
        costs = self.cost_function(estimated_poses, states, actions)
        costs_t = torch.tensor(costs).reshape(-1,1)

        costs_t = costs_t.flatten() # This fixes the error on mppi _compute_rollout_costs, although the documentation says that cost should be a (K,1)
        return costs_t

    def _pack_state_to_tensor(self, state):
        """
        Transform state into a tensor (K, state_size)
        :param state: expected state for the model
        :return: state tensor
        """
        state_t = torch.tensor(state)
        state_t = state_t.reshape(-1, self.state_size)
        return state_t

    def _unpack_state_tensor(self, state_t):
        """
        Transform back the state.
        :param state_t: (K, state_size) tensor
        :return: state -- expected state for the model
        """
        state = state_t
        state = state_t.reshape(-1, *self.original_state_shape)
        return state

    def _unpack_action_tensor(self, action_t):
        action = action_t
        return action

    def _unpack_state_sample(self, state_sample):
        state = None
        state = state_sample['init_imprint']
        # TODO: Extract state from sample
        return state

    def _pack_state_to_sample(self, state, sample_ref):
        sample = copy.deepcopy(sample_ref)
        sample['next_imprint'] = state
        # TODO: Add state to sample
        return sample

    def _get_controller(self):
        self._init_params()
        u_min = torch.zeros((3,)) # TODO: Set from action space
        u_max = torch.tensor([2*np.pi, .15, .15]) # TODO: Set from action space
        lambda_ = 1.0 # TODO: Set
        device = self.model.device
        
        controller = mppi.MPPI(self.dynamics, self.compute_cost, self.state_size, self.noise_sigma,
                               lambda_=lambda_, device=device,
                               num_samples=self.num_samples, horizon=self.horizon, u_min=u_min, u_max=u_max)
        return controller

    def _query_controller(self, state_sample):
        self.sample = state_sample
        state = self._unpack_state_sample(state_sample)
        state_t = self._pack_state_to_tensor(state)
        action = self.controller.command(state_t)
        return action






