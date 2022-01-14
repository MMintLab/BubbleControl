import abc
import torch
import numpy as np
import copy
import tf.transformations as tr
from pytorch_mppi import mppi
import pytorch3d.transforms as batched_trs


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


class BubbleModelMPPIController(object):
    # THis used to inherit from BubbleModelController. In the future abstract out again the general controller stuff.

    def __init__(self, model, env, object_pose_estimator, cost_function, num_samples=100, horizon=3, lambda_=0.01, noise_sigma=None, _noise_sigma_value=0.2):
        self.model = model
        self.env = env
        self.object_pose_estimator = object_pose_estimator
        self.cost_function = cost_function
        self.num_samples = num_samples
        self.horizon = horizon
        self.original_state_shape = None
        self.noise_sigma = noise_sigma
        self._noise_sigma_value = _noise_sigma_value
        self.state_size = None # To be filled with sample information or model information
        self.lambda_ = lambda_
        self.device = self.model.device
        self.u_min, self.u_max = self._get_action_space_limits()

        self.sample = None # Container to share sample across functions
        self.state_size = np.prod(self.model._get_sizes()['imprint']) # TODO: Make more general

        self.controller = self._get_controller()

    def control(self, state_sample):
        action = self._query_controller(state_sample)
        return action

    def dynamics(self, state_t, action_t):
        """
        Compute the dynamics by querying the model
        :param state_t: (K, state_size) tensor
        :param action_t: (K, action_size) tensor
        :return: next_state: (K, state_size) tensor
        """
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
        states = self._unpack_state_tensor(state_t)
        actions = self._unpack_action_tensor(action_t)

        estimated_poses = []
        for i, state_i in enumerate(states):
            # TODO: Add the action pose correction -- consider adding the simulation in the environment.
            state_sample_i = self._pack_state_to_sample(state_i, self.sample)
            estimated_pose_i = self.object_pose_estimator.estimate_pose(state_sample_i)
            estimated_poses.append(estimated_pose_i)
        estimated_poses = np.array(estimated_poses)
        costs = self.cost_function(estimated_poses, states, actions)
        costs_t = torch.tensor(costs).reshape(-1, 1)

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
        self.original_state_shape = self.model.input_sizes['init_imprint']
        self.state_size = np.prod(self.original_state_shape)
        if self.noise_sigma is None:
            self.noise_sigma = self._noise_sigma_value * torch.diag(self.u_max - self.u_min)
        else:
            # convert it to a tensor
            self.noise_sigma = torch.diag(torch.tensor(self.noise_sigma, device=self.device, dtype=torch.float))

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
        controller = mppi.MPPI(self.dynamics, self.compute_cost, self.state_size, self.noise_sigma,
                               lambda_=self.lambda_, device=self.model.device,
                               num_samples=self.num_samples, horizon=self.horizon, u_min=self.u_min, u_max=self.u_max)
        return controller

    def _query_controller(self, state_sample):
        self.sample = state_sample
        state = self._unpack_state_sample(state_sample)
        state_t = self._pack_state_to_tensor(state)
        action = self.controller.command(state_t)
        return action


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
        costs_t = torch.zeros((len(state_t), 1))

        states = self._unpack_state_tensor(state_t)
        actions = self._unpack_action_tensor(action_t)

        state_samples = self._pack_state_to_sample(states, self.sample)
        state_samples = self._action_correction(state_samples, actions)
        estimated_poses = self.object_pose_estimator.estimate_pose(state_samples)
        costs = self.cost_function(estimated_poses, states, actions)
        costs_t = torch.tensor(costs).reshape(-1, 1)

        costs_t = costs_t.flatten() # This fixes the error on mppi _compute_rollout_costs, although the documentation says that cost should be a (K,1)
        return costs_t

    def _pack_state_to_sample(self, state, sample_ref):
        sample = copy.deepcopy(sample_ref)
        batch_size = state.shape[0]
        batch_size = state.shape[0]
        # convert all_tfs to tensors
        sample['all_tfs'] = self._convert_all_tfs_to_tensors(sample['all_tfs'])
        # convert samples to tensors
        batched_sample = batched_tensor_sample(sample, batch_size=batch_size, device=state.device)
        # and repeat the batch size (at least for camera_info_{r,l}['K'], undef_depth_{r,l}, all_tfs
        batched_sample['next_imprint'] = state
        return batched_sample

    def _convert_all_tfs_to_tensors(self, all_tfs):
        """
        :param all_tfs: DataFrame
        :return:
        """
        # TODO: Transform a DF into a dictionary of homogeneous transformations matrices (4x4)
        converted_all_tfs = {}
        parent_frame = all_tfs['parent_frame'][0] # Assume that are all teh same
        child_frames = all_tfs['child_frame']
        converted_all_tfs[parent_frame] = np.eye(4) # Transformation to itself is the identity
        all_poses = all_tfs[['x','y','z','qx','qy','qz','qw']]
        for i, child_frame_i in enumerate(child_frames):
            pose_i = all_poses.iloc[i]
            X_i = tr.quaternion_matrix(pose_i[3:])
            X_i[:3,3] = pose_i[:3]
            converted_all_tfs[child_frame_i] = X_i
        return converted_all_tfs

    def _action_correction(self, state_samples, actions):
        # actions: tensor of shape (N, action_dim)
        state_samples_corrected = state_samples
        # TODO: This is especific for each environment --> TODO: Do it more general
        action_names = ['rotation', 'length', 'grasp_width']
        rotations = actions[..., 0]
        lengths = actions[..., 1]
        grasp_widths = actions[..., 2]
        # Rotation is a rotation about the x axis of the grasp_frame
        # Length is a translation motion of length 'length' of the grasp_frame on the xy med_base plane along the intersection with teh yz grasp frame plane 
        # grasp_width is the width of the  
        # TODO: Implement
        all_tfs = state_samples_corrected['all_tfs'] # Tfs from world frame ('med_base') to the each of teh frame names
        frame_names = all_tfs.keys()

        rigid_ee_frames = ['grasp_frame', 'med_kuka_link_ee', 'wsg50_finger_left', 'pico_flexx_left_link', 'pico_flexx_left_optical_frame', 'pico_flexx_right_link', 'pico_flexx_right_optical_frame']

        wf_X_gf = all_tfs['grasp_frame']
        # Move Gripper: 
        # (move wsg_50_finger_{right,left} along x direction)
        gf_X_fl = get_transformation_matrix(all_tfs, 'grasp_frame', 'wsg50_finger_left')
        gf_X_fr = get_transformation_matrix(all_tfs, 'grasp_frame', 'wsg50_finger_right')
        X_finger_left = torch.eye(4).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double)
        X_finger_right = torch.eye(4).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double)
        current_half_width_l = -gf_X_fl[...,0,3]-0.009
        current_half_width_r = gf_X_fr[...,0,3]-0.009
        X_finger_left[...,0,3] = -(0.5*grasp_widths - current_half_width_l).type(torch.double)
        X_finger_right[...,0,3] = -(0.5*grasp_widths - current_half_width_r).type(torch.double)

        all_tfs = tr_frame(all_tfs, 'wsg50_finger_left', X_finger_left, ['pico_flexx_left_link', 'pico_flexx_left_optical_frame'])
        all_tfs = tr_frame(all_tfs, 'wsg50_finger_right', X_finger_right, ['pico_flexx_right_link', 'pico_flexx_right_optical_frame'])
        # Move Grasp frame on the plane amount 'length; and rotate the Grasp frame along x direction a 'rotation'  amount
        rot_axis = torch.tensor([1,0,0]).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double)
        angle_axis = rotations.unsqueeze(-1).repeat_interleave(3,dim=-1) * rot_axis
        X_gf_rot = torch.eye(4).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double)
        X_gf_rot[...,:3,:3] = batched_trs.axis_angle_to_matrix(angle_axis)# rotation along x axis
        z_axis = torch.tensor([0,0,1]).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double)
        y_dir_gf = torch.tensor([0,-1,0]).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double)
        y_dir_wf = torch.einsum('kij,kj->ki',wf_X_gf[...,:3,:3],  y_dir_gf)
        y_dir_wf_perp = torch.einsum('ki,ki->k',y_dir_wf, z_axis).unsqueeze(-1).repeat_interleave(3,dim=-1)*z_axis
        drawing_dir_wf = y_dir_wf - y_dir_wf_perp
        drawing_dir_wf = drawing_dir_wf / torch.linalg.norm(drawing_dir_wf, dim=1).unsqueeze(-1).repeat_interleave(3,dim=-1) # normalize
        drawing_dir_gf = torch.einsum('kij,kj->ki',torch.linalg.inv(wf_X_gf[..., :3, :3]), drawing_dir_wf)
        trans_gf = lengths.unsqueeze(-1).repeat_interleave(3,dim=-1)*drawing_dir_gf
        X_gf_trans = torch.eye(4).unsqueeze(0).repeat_interleave(actions.shape[0], dim=0).type(torch.double).type(torch.double)
        X_gf_trans[...,:3,3] = trans_gf
        all_tfs = tr_frame(all_tfs, 'grasp_frame', X_gf_trans, rigid_ee_frames)
        all_tfs = tr_frame(all_tfs, 'grasp_frame', X_gf_rot, rigid_ee_frames)

        return state_samples_corrected

def tr_frame(all_tfs, frame_name, X, fixed_frame_names):
    # Apply tf to the frame_name and modify all other tf for the fixed frames to that tf frame
    # all_tfs: dict of tfs
    # frame_name: str for the frame to apply X
    # X (aka fn_X_fn_new) trasformation to be applied along frame_name
    # fixed_frame_names: list of strs containing the names of the frames that we also need to transform because they are rigid to the frame_name frame.
    new_tfs = {}
    w_X_fn = all_tfs[frame_name]
    w_X_fn_new = w_X_fn @ X
    new_tfs[frame_name] = w_X_fn_new
    # Apply the new transformation to all fixed frames
    for ff_i in fixed_frame_names:
        if ff_i == frame_name:
            continue
        w_X_ffi = all_tfs[ff_i]
        fn_X_ffi = get_transformation_matrix(all_tfs, source_frame=frame_name, target_frame=ff_i)
        w_X_ffi_new = w_X_fn_new @ fn_X_ffi
        new_tfs[ff_i] = w_X_ffi_new
    all_tfs.update(new_tfs)
    return all_tfs

def get_transformation_matrix(all_tfs, source_frame, target_frame):
    w_X_sf = all_tfs[source_frame]
    w_X_tf = all_tfs[target_frame]
    sf_X_w = torch.linalg.inv(w_X_sf)
    sf_X_tf = sf_X_w @ w_X_tf
    return sf_X_tf

def batched_tensor_sample(sample, batch_size=None, device=None):
    # sample is a dictionary of
    if device is None:
        device = torch.device('cpu')
    batched_sample = {}
    for k_i, v_i in sample.items():
        if type(v_i) is dict:
            batched_sample_i = batched_tensor_sample(v_i, batch_size=batch_size, device=device)
            batched_sample[k_i] = batched_sample_i
        elif type(v_i) is np.ndarray:
            batched_sample_i = torch.tensor(v_i).to(device)
            if batch_size is not None:
                batched_sample_i = batched_sample_i.unsqueeze(0).repeat_interleave(batch_size, dim=0)
            batched_sample[k_i] = batched_sample_i
        elif type(v_i) in [int, float]:
            batched_sample_i = torch.tensor([v_i]).to(device)
            if batch_size is not None:
                batched_sample_i = batched_sample_i.unsqueeze(0).repeat_interleave(batch_size, dim=0)
            batched_sample[k_i] = batched_sample_i
        elif type(v_i) is torch.Tensor:
            if batch_size is not None:
                batched_sample_i = batched_sample_i.unsqueeze(0).repeat_interleave(batch_size, dim=0)
            batched_sample[k_i] = batched_sample_i.to(device)
        else:
            batched_sample[k_i] = v_i
    return batched_sample
