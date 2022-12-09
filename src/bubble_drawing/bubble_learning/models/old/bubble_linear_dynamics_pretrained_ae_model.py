import torch.nn as nn

from bubble_drawing.bubble_learning.models.old.bubble_dynamics_pretrained_ae_model import BubbleDynamicsPretrainedAEModel, BubbleFullDynamicsPretrainedAEModel


class BubbleLinearDynamicsPretrainedAEModel(BubbleDynamicsPretrainedAEModel):
    """
    Model designed to model the bubbles dynamics.
    Given s_t, and a_t, it produces ∆s, where s_{t+1} = s_t + ∆s
     * Here s_t is composed by:
        - Depth image from each of the bubbles
    * The depth images are embedded into a vector which is later concatenated with the wrench and pose information
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls):
        return 'bubble_linear_dynamics_pretrained_autoencoder_model'

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        action_size = sizes['action']
        dyn_input_size = self.img_embedding_size + action_size
        dyn_output_size = self.img_embedding_size
        dyn_model = nn.Linear(in_features=dyn_input_size, out_features=dyn_output_size, bias=False)
        return dyn_model


class BubbleFullLinearDynamicsPretrainedAEModel(BubbleFullDynamicsPretrainedAEModel):

    @classmethod
    def get_name(cls):
        return 'bubble_full_linear_dynamics_pretrained_autoencoder_model'

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        action_size = sizes['action']
        dyn_input_size = self.img_embedding_size + action_size + sizes['wrench'] + sizes['position'] + sizes['orientation']
        dyn_output_size = self.img_embedding_size + sizes['wrench'] + sizes['position'] + sizes['orientation']
        dyn_model = nn.Linear(in_features=dyn_input_size, out_features=dyn_output_size, bias=False)
        return dyn_model



