import torch
import numpy as np
from tqdm import tqdm


def icp_2d_masked(pc_model, pc_scene, pc_scene_mask, num_iter=30):
    # ICP 2D:
    # pc_scene: (N, n_impr, w, h, n_coords)
    # pc_scene_mask: (N, n_impr, w, h, n_coords)
    # pc_model: (N, n_model_points, n_coords)
    # n_scene_points = n_impr * w * h

    N, n_impr, w, h, n_coords = pc_scene.shape
    pc_scene_r = reshape_pc(pc_scene)
    if len(pc_scene_mask.shape) == len(pc_scene_mask.shape)-1:
        pc_scene_mask = pc_scene_mask.unsqueeze(-1).repeat_interleave(n_coords, dim=-1)  # (N, n_scene_points, n_coords)
    pc_scene_mask_r = reshape_pc(pc_scene_mask)

    R_init = torch.eye(n_coords).unsqueeze(0).repeat_interleave(N, dim=0).type(pc_scene.dtype).to(pc_scene.device)  # (N, num_dims, num_dims)--- init all R as identyty
    t_init = masked_tensor_mean(pc_scene_r.transpose(1, 2), pc_scene_mask_r.transpose(1, 2),
                                start_dim=-1)  # mean of the scene

    for i in tqdm(range(num_iter)):
        R, t = icp_2d_maksed_step(pc_model, pc_scene_r, pc_scene_mask_r, R_init, t_init)
        R_init = R
        t_init = t
    # R: (N, n_coords, n_coords)
    # t: (N, n_coords)
    return R, t


def icp_2d_maksed_step(pc_model, pc_scene, pc_scene_mask, R_init, t_init):
    # pc_model, shape (N, n_model_points, n_coords)
    # pc_scene, shape (N, n_scene_points, n_coords)
    # pc_scene_mask, shape (N, n_scene_points, n_coords) *** Here n_coords dimension is just repeated
    # t_init: (N, n_coords)
    # R_init: (N, n_coords, n_coords)
    # -------------------
    # transform init:
    pc_model_tr = pc_batched_tr(pc_model, R_init, t_init)

    # Estimate correspondences (only masked):
    # compute distances and get minimums
    pc_model_selected = estimate_correspondences_batched(pc_model_tr, pc_scene, pc_scene_mask)

    # Compute new transform
    R_star, t_star = find_best_transform_batched_masked(pc_model_selected, pc_scene, pc_scene_mask)

    return R_star, t_star


def masked_tensor_mean(xs, masks,start_dim=-2):
    # masked mean along the last start_dim dims
    masks_f = masks.flatten(start_dim=start_dim)
    xs_f = xs.flatten(start_dim=start_dim)
    num_occ = torch.sum(masks_f, dim=-1)
    xs_sum = torch.sum(masks_f*xs_f, dim=-1)
    xs_mean_f = xs_sum/num_occ
    xs_mean = xs_mean_f
    return xs_mean


def pc_batched_tr(pc, R, t):
    # pc: (batch_size, n_points, n_coords)
    # R: (batch_size, n_coords, n_coords)
    # t: (batch_size, n_coords)
    pc_rot = torch.einsum('kij,klj->kli', R, pc)  # (batch_size, n_points, n_coords)
    pc_tr = (pc_rot.transpose(0, 1) + t).transpose(0, 1)  # (batch_size, n_points, n_coords)
    return pc_tr


def reshape_pc(pc):
    # pc: (N, n_impr, w, h, n_coords)
    N, n_impr, w, h, n_coords = pc.shape
    # _pc_r = pc.permute(0,1,3,4,2)
    pc_r = pc.reshape(N, n_impr * w * h, n_coords)  # (N, n_impr*w*h, n_coords)
    return pc_r


def estimate_correspondences_batched(a1, a2, a2_mask):
    """
    Return for each point in the scene (a2) the closest point in the model (a1)
    :param a1: (N, n_1_points, n_coords) -- model
    :param a2: (N, n_2_points, n_coords) -- scene
    :param a2_mask: (N, n_2_points, n_coords)
    :return: tensor containing the correspndent model points for each scene points (N, n_2_points, n_coords)
    """
    # Compute distances
    N1, n_1_points, n_coords_1 = a1.shape
    N2, n_2_points, n_coords_2 = a2.shape
    a12 = a1.unsqueeze(1).repeat_interleave(n_2_points, dim=1)
    a21 = a2.unsqueeze(2).repeat_interleave(n_1_points, dim=2)
    dists = torch.sqrt(torch.sum((a12 - a21) ** 2, dim=-1))

    # Get clossest point indxs
    corr_indxs = torch.argmin(dists, axis=-1)  # get a1 index that minimizes distance to a2

    # Apply correspondences
    batch_idxs = torch.arange(0, corr_indxs.shape[0]).unsqueeze(-1).repeat_interleave(n_2_points, dim=-1)
    a_1corr = a1[batch_idxs, corr_indxs, :]
    return a_1corr


def find_best_transform_batched_masked(pc_model, pc_scene, pc_mask):
    # pc_model: (N, n_scene_points, n_coords) -- model
    # pc_scene: (N, n_scene_points, n_coords) -- scene
    # pc_mask: (N, n_scene_points, n_coords)
    N, n_points, n_coords = pc_scene.shape
    #     pc_mask_extended = pc_mask.unsqueeze(-1).repeat(1,1, n_coords)# (N, n_scene_points, n_coords)

    mu_s = masked_tensor_mean(pc_scene.transpose(1, 2), pc_mask.transpose(1, 2), start_dim=-1)  # (N, n_coords)
    mu_m = masked_tensor_mean(pc_model.transpose(1, 2), pc_mask.transpose(1, 2), start_dim=-1)  # (N, n_coords)
    p_s = (pc_scene.transpose(0, 1) - mu_s).transpose(0, 1)  # (N, n_scene_points, n_coords)
    p_m = (pc_model.transpose(0, 1) - mu_m).transpose(0, 1)  # (N, n_scene_points, n_coords)
    # apply mask
    ps_filtered = p_s * pc_mask
    pm_filtered = p_m * pc_mask
    W = torch.einsum('kij,kil->kjl', ps_filtered, pm_filtered)  # (N, n_coords, n_coords)
    U, S, Vh = torch.linalg.svd(W)
    R = U @ Vh
    detR = torch.det(R)  # (N,)
    Vh[:, -1] = Vh[:, -1] * torch.stack([detR] * 2, dim=-1)
    R_star = U @ Vh
    t_star = mu_s - torch.einsum('kij,kj->ki', R_star, mu_m)
    return R_star, t_star


