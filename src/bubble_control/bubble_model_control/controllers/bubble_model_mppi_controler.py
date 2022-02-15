import abc
import torch
import numpy as np
import copy
import tf.transformations as tr
from pytorch_mppi import mppi
import pytorch3d.transforms as batched_trs

from bubble_control.bubble_model_control.controllers.bubble_controller_base import BubbleModelController
from bubble_control.bubble_model_control.aux.bubble_model_control_utils import batched_tensor_sample, get_transformation_matrix, tr_frame, convert_all_tfs_to_tensors


class BubbleModelMPPIController(BubbleModelController):
    # THis used to inherit from BubbleModelController. In the future abstract out again the general controller stuff.

    def __init__(self, model, env, object_pose_estimator, cost_function, action_model, state_trs=None, num_samples=100, horizon=3, lambda_=0.01, noise_sigma=None, _noise_sigma_value=0.2):
        self.action_model = action_model
        self.num_samples = num_samples
        self.horizon = horizon
        super().__init__(model, env, object_pose_estimator, cost_function, state_trs=state_trs)
        self.u_mu = None
        self.noise_sigma = noise_sigma
        self._noise_sigma_value = _noise_sigma_value
        self.state_size = None # To be filled with sample information or model information
        self.lambda_ = lambda_
        self.device = self.model.device
        self.u_min, self.u_max = self._get_action_space_limits()
        self.U_init = None # Initial trajectory. We initialize it as the mean of the action space. Actions will be drawin as a gaussian noise added to this values.

        self.sample = None # Container to share sample across functions
        self.original_state_shape = None
        self.state_size = None
        self.controller = None # controller not initialized yet
        self.action_container = self._get_action_container()
        
    def _get_action_container(self):
        action_container, _ = self.env.get_action()
        return action_container

    def control(self, state_sample):
        # pack the action to the env format
        action_raw = super().control(state_sample).detach().cpu().numpy()
        for i, (k, v) in enumerate(self.action_container.items()):
            self.action_container[k] = action_raw[i]
            if i+1 >= action_raw.shape[-1]:
                break
        return self.action_container

    def dynamics(self, state_t, action_t):
        """
        Compute the dynamics by querying the model
        :param state_t: (K, state_size) tensor
        :param action_t: (K, action_size) tensor
        :return: next_state: (K, state_size) tensor
        """
        state = self._unpack_state_tensor(state_t)
        action = self._unpack_action_tensor(action_t)
        output = self.model(*state, action)
        next_state = self._expand_output_to_state(output, state, action)
        next_state_t = self._pack_state_to_tensor(next_state)
        return next_state_t

    def compute_cost(self, state_t, action_t):
        """
        Compute the dynamics
        :param state: (K, state_size) tensor
        :param action: (K, action_size) tensor
        :return: cost: (K, 1) tensor
        """
        states = self._unpack_state_tensor(state_t)
        actions = self._unpack_action_tensor(action_t)

        estimated_poses = []
        for i, state_i in enumerate(states):
            state_sample_i = self._pack_state_to_sample(state_i, self.sample)
            # TODO: Add action correction?
            estimated_pose_i = self.object_pose_estimator.estimate_pose(state_sample_i)
            estimated_poses.append(estimated_pose_i)
        estimated_poses = np.array(estimated_poses)
        costs = self.cost_function(estimated_poses, states, states, actions)
        if not torch.is_tensor(costs):
            costs_t = torch.tensor(costs)
        else:
            costs_t = costs

        costs_t = costs_t.flatten() # This fixes the error on mppi _compute_rollout_costs, although the documentation says that cost should be a (K,1)
        return costs_t

    def _get_action_space_limits(self):
        # Process the action space to get the u_min and u_max
        low_limits = []
        high_limits = []
        for action_k, action_s in self.env.action_space.items():
            low_limits.append(action_s.low.flatten())
            high_limits.append(action_s.high.flatten())
        u_min = np.concatenate(low_limits)
        u_max = np.concatenate(high_limits)
        u_min = torch.tensor(u_min, device=self.device, dtype=torch.float)
        u_max = torch.tensor(u_max, device=self.device, dtype=torch.float)
        return u_min, u_max

    def _init_params(self):
        # TODO: Make more general
        # state_size is the sum of sizes of all items in the state
        if self.noise_sigma is None or self.noise_sigma.shape[0] != self.u_max.shape[0]:
            eps = 1e-7
            self.noise_sigma = self._noise_sigma_value * torch.diag(self.u_max - self.u_min + eps)
        else:
            # convert it to a square tensor
            self.noise_sigma = torch.diag(torch.tensor(self.noise_sigma, device=self.device, dtype=torch.float))
        if self.u_mu is None or self.u_mu.shape[-1] != self.u_max.shape[0]:
            self.u_mu = 0.5 * (self.u_max + self.u_min)
        if self.U_init is None or self.U_init.shape[-1] != self.u_max.shape[0]:
            self.U_init = self.u_mu.unsqueeze(0).repeat_interleave(self.horizon, dim=0)

    def _pack_state_to_tensor(self, state):
        """
        Transform state into a tensor (K, state_size)
        :param state: expected state for the model
        :return: state tensor
        """
        imprint = state[0]
        if not torch.is_tensor(imprint):
            state_t = torch.tensor(imprint)
        else:
            state_t = imprint
        state_t = state_t.reshape(-1, self.state_size)
        return state_t

    def _unpack_state_tensor(self, state_t):
        """
        Transform back the state.
        :param state_t: (K, state_size) tensor
        :return: state -- expected state for the model
        """
        imprint = state_t.reshape(-1, *self.original_state_shape)
        state = (imprint,)
        return state

    def _unpack_action_tensor(self, action_t):
        action = action_t
        return action

    def _unpack_state_sample(self, state_sample):
        # Extract state from sample
        imprint = state_sample['init_imprint']
        state = (imprint, )
        return state

    def _pack_state_to_sample(self, state, sample_ref):
        # Add state to sample
        imprint = state[0]
        sample = copy.deepcopy(sample_ref)
        sample['next_imprint'] = imprint
        return sample

    def _expand_output_to_state(self, output, state, action):
        expanded_state = (output,)
        return expanded_state

    def _get_controller(self):
        self._init_params()
        controller = mppi.MPPI(self.dynamics, self.compute_cost, self.state_size, self.noise_sigma,
                               lambda_=self.lambda_, device=self.model.device,
                               num_samples=self.num_samples, horizon=self.horizon, u_min=self.u_min, u_max=self.u_max, u_init=self.u_mu, U_init=self.U_init, noise_abs_cost=True)
        return controller

    def _query_controller(self, state_sample):
        if not self.controller:
            # Initialize the controller
            self.original_state_shape = state_sample['init_imprint'].shape
            self.state_size = np.prod(self.original_state_shape)
            self.controller = self._get_controller()
        self.sample = state_sample
        state = self._unpack_state_sample(state_sample)
        state_t = self._pack_state_to_tensor(state)
        action = self.controller.command(state_t)
        return action

    def _action_correction(self, state_samples, actions):
        # actions: tensor of shape (N, action_dim)
        state_samples_corrected = self.action_model(state_samples, actions)
        return state_samples_corrected


class BubbleModelMPPIBatchedController(BubbleModelMPPIController):
    """
    The object_pose_estimator admits batched samples
    """


    def compute_cost(self, state_t, action_t):
        """
        Compute the dynamics
        :param state: (K, state_size) tensor
        :param action: (K, action_size) tensor
        :return: cost: (K, 1) tensor
        """
        states = self._unpack_state_tensor(state_t)
        actions = self._unpack_action_tensor(action_t)
        state_samples = self._pack_state_to_sample(states, self.sample)
        prev_state_samples = state_samples.copy()
        state_samples = self._action_correction(state_samples, actions)
        estimated_poses = self.object_pose_estimator.estimate_pose(state_samples)
        costs = self.cost_function(estimated_poses, state_samples, prev_state_samples, actions)
        if not torch.is_tensor(costs):
            costs_t = torch.tensor(costs)
        else:
            costs_t = costs
        costs_t = costs_t.flatten() # This fixes the error on mppi _compute_rollout_costs, although the documentation says that cost should be a (K,1)
        return costs_t

    def _pack_state_to_sample(self, state, sample_ref):
        sample = sample_ref.copy() # No copy
        imprint = state[0]
        batch_size = imprint.shape[0]
        # convert all_tfs to tensors
        sample['all_tfs'] = self._convert_all_tfs_to_tensors(sample['all_tfs'])
        # convert samples to tensors
        batched_sample = batched_tensor_sample(sample, batch_size=batch_size, device=imprint.device)
        # and repeat the batch size (at least for camera_info_{r,l}['K'], undef_depth_{r,l}, all_tfs
        batched_sample['next_imprint'] = imprint
        return batched_sample

    def _convert_all_tfs_to_tensors(self, all_tfs):
        """
        :param all_tfs: DataFrame
        :return:
        """
        converted_all_tfs = convert_all_tfs_to_tensors(all_tfs)
        return converted_all_tfs




