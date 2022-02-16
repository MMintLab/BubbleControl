import abc
import torch
import numpy as np
import copy
import tf.transformations as tr
from pytorch_mppi import mppi
import pytorch3d.transforms as batched_trs
import pdb
from bubble_control.bubble_model_control.controllers.bubble_controller_base import BubbleModelController
from bubble_control.bubble_model_control.aux.bubble_model_control_utils import batched_tensor_sample, get_transformation_matrix, tr_frame, convert_all_tfs_to_tensors


def to_tensor(x, **kwargs):
    if not torch.is_tensor(x):
        x_t = torch.tensor(x, **kwargs)
    else:
        x_t = x
    return x_t

def default_grasp_pose_correction(position, orientation, action):
    return position, orientation

class BubbleModelMPPIController(BubbleModelController):
    """
    Batched controller with a batched pose estimation
    """
    def __init__(self, model, env, object_pose_estimator, cost_function, action_model, grasp_pose_correction=default_grasp_pose_correction, 
                 state_trs=None, num_samples=100, horizon=3, lambda_=0.01, noise_sigma=None, _noise_sigma_value=0.2):
        self.action_model = action_model
        self.grasp_pose_correction = grasp_pose_correction
        self.num_samples = num_samples
        self.horizon = horizon
        super().__init__(model, env, object_pose_estimator, cost_function, state_trs=state_trs)
        self.u_mu = None
        self.noise_sigma = noise_sigma
        self._noise_sigma_value = _noise_sigma_value
        self.lambda_ = lambda_
        self.device = self.model.device
        self.action_container = self._get_action_container()
        self.u_min, self.u_max = self._get_action_space_limits()
        self.U_init = None # Initial trajectory. We initialize it as the mean of the action space. Actions will be drawin as a gaussian noise added to this values.
        self.input_keys = self.model.get_input_keys()
        self.state_keys = self.model.get_state_keys()
        self.model_output_keys = self.model.get_model_output_keys()
        self.next_state_map = self.model.get_next_state_map()
        self.state_size = None
        self.original_state_shape = None
        self.sample = None # Container to share sample across functions
        self.controller = None # controller not initialized yet
        self.action_space = self.env.action_space
        
    def _get_action_container(self):
        action_container, _ = self.env.get_action()
        return action_container

    @abc.abstractmethod
    def _get_state_keys(self):
        pass

    @abc.abstractmethod
    def _get_model_output_keys(self):
        pass

    @abc.abstractmethod
    def _get_next_state_map(self):
        pass

    def compute_cost(self, state_t, action_t):
        """
        Compute the dynamics
        :param state: (K, state_size) tensor
        :param action: (K, action_size) tensor
        :return: cost: (K, 1) tensor
        """
        # State_t is already next state but state_sample still has old tfs until we apply pack_state_to_sample and action_correction
        states = self._unpack_state_tensor(state_t)
        actions = self._unpack_action_tensor(action_t)
        state_samples = self._pack_state_to_sample(states, self.sample)
        prev_state_samples = state_samples.copy()
        state_samples = self._action_correction(state_samples, actions) # apply the action model
        estimated_poses = self._estimate_poses(state_samples, actions)
        costs = self.cost_function(estimated_poses, state_samples, prev_state_samples, actions)
        costs_t = to_tensor(costs)
        costs_t = costs_t.flatten()  # This fixes the error on mppi _compute_rollout_costs, although the documentation says that cost should be a (K,1)
        return costs_t

    def _estimate_poses(self, state_samples, actions):
        estimated_poses = self.object_pose_estimator.estimate_pose(state_samples) # Batched case
        return estimated_poses

    def _pack_state_to_tensor(self, state):
        """
        Transform state into a tensor (K, state_size)
        :param state: tuple of tensors representing the state (expected input to the model)
        :return: state tensor
        """
        state_t = [to_tensor(s).flatten(start_dim=1) for s in state]
        state_t = torch.cat(state_t, dim=-1)
        return state_t

    def _pack_state_to_sample(self, state, sample_ref):
        """
        Convert the state to a sample
        :param state:
        :param sample_ref:
        :return: sample containing the state
        """
        sample = sample_ref.copy()  # No copy
        batch_size = state[0].shape[0]
        device = state[0].device
        # convert all_tfs to tensors
        sample['all_tfs'] = self._convert_all_tfs_to_tensors(sample['all_tfs'])
        # convert samples to tensors
        batched_sample = batched_tensor_sample(sample, batch_size=batch_size, device=device)
        # and repeat the batch size (at least for camera_info_{r,l}['K'], undef_depth_{r,l}, all_tfs

        # put the state to the sample
        for i, key in enumerate(self.state_keys):
            state_i = state[i]
            if key in self.next_state_map:
                new_key = self.next_state_map[key]
            else:
                new_key = key
            batched_sample[new_key] = state_i
        return batched_sample

    def _unpack_state_tensor(self, state_t):
        """
        Transform back the state.
        :param state_t: (K, state_size) tensor
        :return: state -- expected state for the model
        """
        flattened_sizes = self._get_flattened_state_sizes()
        state_split = torch.split(state_t, flattened_sizes, dim=-1)
        state = []
        for i, (k, original_size_i) in enumerate(self.original_state_shape.items()):
            state_i = state_split[i]
            state_i_unpacked = state_i.reshape(-1, *original_size_i)
            state.append(state_i_unpacked)
        state = tuple(state)
        return state

    def _unpack_state_sample(self, state_sample):
        """
        Extract the state from a sample
        :param state_sample:
        :return: state -- tuple of tensors
        """
        state = []
        for key in self.state_keys:
            state.append(state_sample[key])
        state = tuple(state)
        return state

    def _expand_output_to_state(self, output, state, action):
        """
        Given
        :param output: Model output (can be a subset of the state)
        :param state: Input state
        :param action: Input action
        :return: next state (model predicted)
        """
        expanded_state = []
        for i, k in enumerate(self.state_keys):
            if k in self.model_output_keys:
                output_indx = self.model_output_keys.index(k)
                expanded_state.append(output[output_indx])
            else:
                expanded_state.append(state[k])
        position_idx = self.state_keys.index('position')
        orientation_idx = self.state_keys.index('orientation')
        expanded_state[position_idx], expanded_state[orientation_idx] = self.grasp_pose_correction(expanded_state[position_idx],
                                                                                                expanded_state[orientation_idx],
                                                                                                action)
        return tuple(expanded_state)
    
    def _extract_input_from_state(self, state):
        """
        Given
        :param state: Input state
        :return: model_input (state with just the input keys)
        """
        model_input = []
        for i, k in enumerate(self.state_keys):
            if k in self.input_keys:
                model_input.append(state[i])
        return tuple(model_input)

    def _unpack_action_tensor(self, action_t):
        action = action_t
        return action

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
        model_input = self._extract_input_from_state(state)
        output = self.model(*state, action)
        next_state = self._expand_output_to_state(output, state, action)
        next_state_t = self._pack_state_to_tensor(next_state)
        return next_state_t

    def _get_action_container(self):
        action_container, _ = self.env.get_action()
        return action_container

    def _get_action_space_limits(self):
        # Process the action space to get the u_min and u_max
        low_limits = []
        high_limits = []
        for action_k, action_s in self.action_space.items():
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

    def _get_controller(self):
        self._init_params()
        controller = mppi.MPPI(self.dynamics, self.compute_cost, self.state_size, self.noise_sigma,
                               lambda_=self.lambda_, device=self.model.device,
                               num_samples=self.num_samples, horizon=self.horizon, u_min=self.u_min, u_max=self.u_max, u_init=self.u_mu, U_init=self.U_init, noise_abs_cost=True)
        return controller

    def _query_controller(self, state_sample):
        if not self.controller:
            # Initialize the controller
            self.original_state_shape = self._get_original_state_shape(state_sample)
            self.state_size = self._get_state_size()
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

    def _convert_all_tfs_to_tensors(self, all_tfs):
        """
        :param all_tfs: DataFrame
        :return:
        """
        converted_all_tfs = convert_all_tfs_to_tensors(all_tfs)
        return converted_all_tfs

    def _get_original_state_shape(self, state_sample):
        original_state_shape = {}
        for key in self.state_keys:
            state_k = state_sample[key]
            original_state_shape[key] = state_k.shape[1:] # remove the batch size
        return original_state_shape

    def _get_state_size(self):
        flattened_state_shapes = self._get_flattened_state_sizes()
        state_size = np.sum(flattened_state_shapes)
        return state_size

    def _get_flattened_state_sizes(self):
        flattened_state_sizes= [np.prod(size_i) for k, size_i in self.original_state_shape.items()]
        return flattened_state_sizes



