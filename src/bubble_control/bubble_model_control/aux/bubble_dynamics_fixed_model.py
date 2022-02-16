import torch
import torch.nn as nn


class BubbleDynamicsFixedModel(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = torch.device(device)

    @classmethod
    def get_name(cls):
        return 'bubble_dynamics_fixed_model'

    @property
    def name(self):
        return self.get_name()

    def forward(self, imprint, wrench, object_model, pos, ori, action):
        imprint_next = imprint # Assume that the object will stay at the same place and therefore the imprint will be the same
        wrench_next = wrench
        return imprint_next, wrench_next

