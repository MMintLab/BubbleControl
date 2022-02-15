import torch.nn as nn

from bubble_control.bubble_learning.models.bubble_dynamics_model import BubbleDynamicsModel


class BubbleLinearDynamicsModel(BubbleDynamicsModel):
    @classmethod
    def get_name(cls):
        return 'bubble_linear_dynamics_model'

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        action_size = sizes['action']
        dyn_input_size = self.img_embedding_size + action_size + sizes['wrench'] + sizes['position'] + sizes['orientation']
        dyn_output_size = self.img_embedding_size + sizes['wrench'] + sizes['position'] + sizes['orientation']
        dyn_model = nn.Linear(in_features=dyn_input_size, out_features=dyn_output_size, bias=False)
        return dyn_model


