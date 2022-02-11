import torch

from bubble_control.bubble_learning.models.old.bubble_dynamics_residual_model import BubbleDynamicsResidualModel
from bubble_control.bubble_learning.models.bubble_autoencoder import BubbleAutoEncoderModel
from bubble_control.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from bubble_control.bubble_learning.models.old.bubble_pca_dynamics_residual_model import BubblePCADynamicsResidualModel

from bubble_control.bubble_learning.train_files.parsed_trainer import ParsedTrainer


class MNISTAutoencoderTrainer(ParsedTrainer):

    def __init__(self, Model, default_args=None, default_types=None):
        super().__init__(Model, None, default_args=default_args, default_types=default_types)


    def _add_dataset_args(self, parser):
        pass

    def _get_dataset(self):
        train_dataset =
        val_dataset =

    def _get_train_val_data(self):






if __name__ == '__main__':



    # params:
    default_params = {
        'data_name' : '/home/mik/',
        # 'batch_size' : None,
        # 'val_batch_size' : None,
        # 'max_epochs' : 500,
        # 'train_fraction' : 0.8,
        # 'lr' : 1e-4,
        # 'seed' : 0,
        # 'activation' : 'relu',
        'img_embedding_size' : 20,
        # 'encoder_num_convs' : 3,
        # 'decoder_num_convs' : 3,
        # 'encoder_conv_hidden_sizes' : None,
        # 'decoder_conv_hidden_sizes' : None,
        # 'ks' : 3,
        # 'num_fcs' : 3,
        # 'num_encoder_fcs' : 2,
        # 'num_decoder_fcs' : 2,
        # 'fc_h_dim' : 100,
        # 'skip_layers' : None,
        #
        # 'num_workers' : 8,

        'model': BubbleDynamicsResidualModel.get_name(),
        'wrench_frame' : 'med_base',
        'tf_frame' : 'grasp_frame',
        'dtype' : torch.float32,
        'transformation' : trs

    }
    default_types = {
        'batch_size': int,
        'val_batch_size': int
    }
    Model = [BubbleDynamicsResidualModel, BubbleAutoEncoderModel, BubblePCADynamicsResidualModel]
    Dataset = BubbleDrawingDataset
    parsed_trainer = ParsedTrainer(Model, Dataset, default_args=default_params, default_types=default_types)

    parsed_trainer.train()