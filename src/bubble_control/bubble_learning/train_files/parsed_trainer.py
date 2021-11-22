import argparse
import torch
import inspect
import copy
import os
import pytorch_lightning as pl
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import random_split


class ParsedTrainer(object):
    """
    Class for training one or multiple
    All options are provided on the commandline using argparse
    """

    def __init__(self, Model, Dataset, default_args=None, default_types=None, gpu=True):
        self.gpu = gpu
        self.default_args = default_args
        self.models_dict = self._get_models(Model)
        self.datasets_dict = self._get_datasets(Dataset)
        self.default_types = self._get_default_types(default_types)
        self.parser = self._get_parser()

        self.args = self._parse_args()
        self.dataset = self._get_dataset()
        self.train_loader, self.val_loader = self._get_loaders()
        self.model = self._get_model()
    
    def _get_models(self, Model):
        model_list = list(Model)
        model_names = [Model.name for Model in model_list]
        model_dict = dict(zip(model_names, model_list))
        return model_dict
    
    def _get_datasets(self, dataset):
        dataset_list = list(dataset)
        dataset_names = [d.name for d in dataset_list]
        dataset_dict = dict(zip(dataset_names, dataset_list))
        return dataset_dict

    def _get_common_params(self):
        common_params = {
            'batch_size': None,
            'val_batch_size': None,
            'max_epochs': 500,
            'train_fraction': 0.8,
            'lr': 1e-4,
            'seed': 0,
            'num_workers': 8,
        }
        return common_params

    def _get_default_types(self, default_types=None):
        default_types_base = {
            'batch_size': int,
            'val_batch_size': int
        }
        default_types_out = copy.deepcopy(default_types_base)
        if default_types is not None:
            # Combine them
            for k, v in default_types.items():
                default_types_out[k] = v
        return default_types_out

    def _get_parser(self):
        parser_name = '{}_parser'.format(self.__class__.__name__.lower())
        parser = argparse.ArgumentParser(parser_name)
        self._add_common_args(parser)
        self._add_dataset_args(parser)
        self._add_model_args(parser)
        return parser

    def _add_common_args(self, parser):
        # Default args common for all models:
        parser.add_argument('model', type=str, default=list(self.models_dict.keys())[0], help='used to select the model name. (Possible options: {})'.format(self.models_dict.keys()))
        parser.add_argument('dataset', type=str, default=list(self.models_dict.keys())[0], help='used to select the model name. (Possible options: {})'.format(self.models_dict.keys()))
        common_params = self._get_common_params()
        for k, v in common_params.items():
            self._add_argument(parser, arg_name=k, default_value=v)

    def _add_dataset_args(self, parser):
        parser.add_argument('')
        # TODO: Implement


    def _add_model_args(self, parser):
        subparsers = parser.add_subparsers(dest='model_name')
        for i, Model_i in enumerate(self.models_dict):
            model_name_i = Model_i.get_name()
            subparser_i = subparsers.add_parser(model_name_i)
            # add Model_i arguments:
            model_i_signature = inspect.signature(Model_i)
            for param_name, param_i in model_i_signature.parameters.items():
                if param_i.default is param_i.empty:
                    # No default value cse
                    pass
                    # TODO
                else:
                    self._add_argument(subparser_i, param_name, param_i, extra_help=' - ({})'.format(model_name_i))

    def _add_argument(self, parser, arg_name, default_value, extra_help=None):
        # If we have to consider special cases (types, multiple args...), extend this method.
        if arg_name in self.default_types:
            param_type = self.default_types[arg_name]
        else:
            param_type = type(default_value)# get the same param type as the parameter
        if arg_name in self.default_args:
            default_value = self.default_args[arg_name]
            
        help_str = '{}'.format(arg_name)
        if extra_help is not None:
            help_str += ' {}'.format(extra_help)
        parser.add_argument('--{}'.format(arg_name), default=default_value, type=param_type, help=help_str)

    def _parse_args(self):
        args = self.parser.parse_args()
        args = vars(args) # convert it from a namespace to a dict
        return args
    
    def _get_dataset(self):
        Dataset = self.datasets_dict[self.args['dataset_name']]
        # TODO: Add dataset args
        # # Get the specific parsed parameters
        dataset_arg_names = list(inspect.signature(Dataset).parameters.keys())
        dataset_args = {}
        # TODO: Add dataset specific arguments to be logged
        # for k, v in self.args.items():
        #     if k in dataset_arg_names:
        #         dataset_args[k] = v
        # # Add dataset params
        dataset_args['data_name'] = self.args['data_name']
        model = self._init_dataset(Dataset, dataset_args)
        return model

    def _init_dataset(self, Dataset, dataset_args):
        # TODO: Override this if our model has special inputs
        dataset = Dataset(**dataset_args)
        return dataset

    def _get_loaders(self):
        train_size = int(len(self.dataset) * self.args['train_fraction'])
        val_size = len(self.dataset) - train_size
        train_data, val_data = random_split(self.dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(self.args['seed']))
        batch_size = self.args['batch_size']
        val_batch_size = self.args['val_batch_size']
        if batch_size is None:
            batch_size = train_size
        if val_batch_size is None:
            val_batch_size = val_size
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=self.args['num_workers'],
                                  drop_last=True)
        val_loader = DataLoader(val_data, batch_size=val_batch_size, num_workers=self.args['num_workers'],
                                drop_last=True)

        sizes = self.dataset.get_sizes()

        dataset_params = {
            'batch_size': batch_size,
            'data_name': self.args['data_name'],
            'num_train_samples': len(train_data),
            'num_val_samples': len(val_data),
        }
        # log important information
        self.args['datasaet_params'] = dataset_params
        self.args['sizes'] = sizes

        return train_loader, val_loader

    def _get_model(self):
        Model = self.models_dict[self.args['model']]  # Select the model class

        # Get the specific parsed parameters
        model_arg_names = list(inspect.signature(Model).parameters.keys())
        model_args = {}
        for k, v in self.args.items():
            if k in model_arg_names:
                model_args[k] = v
        # Add dataset params
        model_args['dataset_params'] = self.args['dataset_params']
        # TODO: Add dataset specific arguments to be logged
        model = self._init_model(Model, model_args)
        
        return model 
    
    def _init_model(self, Model, model_args):
        # TODO: Override this if our model has special inputs
        model = Model(**model_args)
        return model

    def _get_logger(self):
        logger = TensorBoardLogger(os.path.join(self.args['data_name'], 'tb_logs'), name=self.model.name)
        return logger

    def train(self):
        logger = self._get_logger()
        gpus = 0
        if torch.cuda.is_available() and self.gpu:
            gpus = 1
        trainer = pl.Trainer(gpus=gpus, max_epochs=self.args['max_epochs'], logger=logger)
        trainer.fit(self.model, self.train_loader, self.val_loader)