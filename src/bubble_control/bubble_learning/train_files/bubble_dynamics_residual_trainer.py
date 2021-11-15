import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split

from bubble_control.bubble_learning.models.bubble_dynamics_residual_model import BubbleDynamicsResidualModel
from bubble_pivoting.datasets.bubble_pivoting_dataset import BubblePivotingDataset
from bubble_control.bubble_learning.aux.orientation_trs import QuaternionToAxis



if __name__ == '__main__':

    # params:
    data_name = '/home/mireia/Documents/research/bubble_pivoting_data'

    batch_size = 5
    max_epochs = 500
    train_fraction = 0.8
    lr = 1e-4
    seed = 0
    activation = 'relu'
    img_embedding_size = 20
    encoder_num_convs = 3
    decoder_num_convs = 3
    encoder_conv_hidden_sizes = None
    decoder_conv_hidden_sizes = None
    ks = 3
    num_fcs = 3
    num_encoder_fcs = 2
    num_decoder_fcs = 2
    fc_h_dim = 100
    skip_layers = None


    # Load dataset
    num_workers = 8
    trs = [QuaternionToAxis()]
    dataset = BubblePivotingDataset(data_name=data_name, wrench_frame='med_base', tf_frame='grasp_frame', dtype=torch.float32, transformation=trs)
    train_size = int(len(dataset) * train_fraction)
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size],  generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, drop_last=True)

    sizes = dataset.get_sizes()

    dataset_params = {
        'batch_size': batch_size,
        'data_name': data_name,
        'num_train_samples': len(train_data),
        'num_val_samples': len(val_data),
        'wrench_frame': dataset.wrench_frame,
        'tf_frame': dataset.tf_frame,
    }

    model = BubbleDynamicsResidualModel(input_sizes=sizes,
                                        img_embedding_size=img_embedding_size,
                                        encoder_num_convs=encoder_num_convs,
                                        decoder_num_convs=decoder_num_convs,
                                        encoder_conv_hidden_sizes=encoder_conv_hidden_sizes,
                                        decoder_conv_hidden_sizes=decoder_conv_hidden_sizes,
                                        ks=ks,
                                        num_fcs=num_fcs,
                                        num_encoder_fcs=num_encoder_fcs,
                                        num_decoder_fcs=num_decoder_fcs,
                                        fc_h_dim=fc_h_dim,
                                        skip_layers=skip_layers,
                                        lr=lr,
                                        dataset_params=dataset_params,
                                        activation=activation
                                        )

    logger = TensorBoardLogger('tb_logs', name=model.name)

    # Train the model
    gpus = 0
    # if torch.cuda.is_available():
    #     gpus = 1
    trainer = pl.Trainer(gpus=gpus, max_epochs=max_epochs, logger=logger)
    trainer.fit(model, train_loader, val_loader)