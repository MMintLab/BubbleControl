import numpy as np
import os
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import pytorch_lightning as pl
import abc
import torchvision
import pytorch3d.transforms as batched_trs
from matplotlib import cm


def get_imprint_grid(batched_imprints, cmap='jet', border_pixels=5):
    # reshape the batched_img to have the same imprints one above the other
    batched_imprint_reshaped = batched_imprints.reshape(*batched_imprints.shape[:1], -1, *batched_imprints.shape[3:])  # (batch_size, 2*W, H)

    grid_img = get_batched_image_grid(batched_imprint_reshaped, cmap=cmap, border_pixels=border_pixels)
    return grid_img


def get_batched_image_grid(batched_img, cmap='jet', border_pixels=5):
    # reshape the batched_img to have the same imprints one above the other
    batched_img = batched_img.detach().cpu()
    # Add padding
    batched_img_padded = F.pad(input=batched_img,
                               pad=(border_pixels, border_pixels, border_pixels, border_pixels),
                               mode='constant',
                               value=0)
    batched_img_cmap = cmap_tensor(batched_img_padded, cmap=cmap)  # size (..., w,h, 3)
    num_dims = len(batched_img_cmap.shape)
    grid_input = batched_img_cmap.permute(*np.arange(num_dims - 3), -1, -3, -2)
    grid_img = torchvision.utils.make_grid(grid_input)
    return grid_img


def cmap_tensor(self, img_tensor, cmap='jet'):
    cmap = cm.get_cmap(cmap)
    mapped_img_ar = cmap(img_tensor / torch.max(img_tensor))  # (..,w,h,4)
    mapped_img_ar = mapped_img_ar[..., :3]  # (..,w,h,3) -- get rid of the alpha value
    mapped_img = torch.tensor(mapped_img_ar).to(self.device)
    return mapped_img