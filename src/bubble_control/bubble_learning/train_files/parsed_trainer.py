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

from bubble_utils.bubble_datasets.bubble_dataset_base import DatasetBase, BubbleDatasetBase
from bubble_utils.bubble_datasets.dataset_transformed import transform_dataset
from bubble_control.bubble_learning.aux.dataframe_tr import SplitDataFramesTr

class ParsedTrainer(object):
    """
    Class for training one or multiple
    All options are provided on the commandline using argparse
    """

    def __init__(self, Model, Dataset, default_args=None, default_types=None):
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
        try:
            model_list = list(Model)
        except TypeError as te:
            Model = [Model]
            model_list = list(Model)
        model_names = [Model.get_name() for Model in model_list]
        model_dict = dict(zip(model_names, model_list))
        return model_dict
    
    def _get_datasets(self, Dataset):
        try:
            dataset_list = list(Dataset)
        except TypeError as te:
            Dataset = [Dataset]
            dataset_list = list(Dataset)
        dataset_names = [d.get_name() for d in dataset_list]
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
        self._add_dataset_args(parser)
        self._add_common_args(parser)
        self._add_model_args(parser)
        return parser

    def _add_common_args(self, parser):
        # Default args common for all models:
        common_params = self._get_common_params()
        for k, v in common_params.items():
            self._add_argument(parser, arg_name=k, default_value=v)
        args = vars(parser.parse_known_args()[0])
        for k,v in self.default_args.items():
            if k not in args:
                self._add_argument(parser, k, v)
        # Add no_gpu option
        parser.add_argument('--no_gpu', action='store_true', help='avoid using the gpu even when it is available')

    def _add_dataset_args(self, parser):
        # TODO: Try to add it as another subparser, but it looks like only one subparser is allowed
        # subparsers = parser.add_subparsers(dest='dataset_name', help='used to select the model name. (Possible options: {})'.format(self.models_dict.keys()))
        parser.add_argument('dataset_name', type=str, default=list(self.datasets_dict.keys())[0], help='used to select the dataset name. (Possible options: {})'.format(self.datasets_dict.keys()))
        for dataset_name, Dataset_i in self.datasets_dict.items():
            arguments_i = self._get_dataset_constructor_arguments(Dataset_i)
            # TODO: Implement by adding the defua

    def _add_model_args(self, parser):
        subparsers = parser.add_subparsers(dest='model_name',help='used to select the model name. (Possible options: {})'.format(self.models_dict.keys()))
        # subparsers = parser.add_subparsers('model_name', type=str, default=list(self.models_dict.keys())[0], help='used to select the model name. (Possible options: {})'.format(self.models_dict.keys()))
        for model_name, Model_i in self.models_dict.items():
            model_name_i = Model_i.get_name()
            subparser_i = subparsers.add_parser(model_name_i)
            # add Model_i arguments:
            model_constructor_args = self._get_model_constructor_arguments(Model_i)
            for param_name, param_i in model_constructor_args.items():
                if False:
                # if param_i is inspect._empty:
                    # No default value cse
                    pass
                else:
                    self._add_argument(subparser_i, param_name, param_i, extra_help=' - ({})'.format(model_name_i))

    def _add_argument(self, parser, arg_name, default_value, extra_help=None):
        # If we have to consider special cases (types, multiple args...), extend this method.
        if arg_name in self.default_args:
            default_value = self.default_args[arg_name]
            param_type = type(default_value)
            if param_type == list and arg_name in self.default_types:
                param_type = self.default_types[arg_name]
        elif arg_name in self.default_types:
            param_type = self.default_types[arg_name]
        else:
            param_type = type(default_value)# get the same param type as the parameter
        help_str = '{}'.format(arg_name)
        if extra_help is not None:
            help_str += ' {}'.format(extra_help)
        if type(default_value) == list:
            nargs='+'
        else:
            nargs=None
        parser.add_argument('--{}'.format(arg_name), default=default_value, type=param_type, help=help_str, nargs=nargs)

    def _parse_args(self):
        args = self.parser.parse_args()
        args = vars(args) # convert it from a namespace to a dict
        return args
    
    def _get_dataset(self):
        Dataset = self.datasets_dict[self.args['dataset_name']]
        # TODO: Add dataset args
        # # Get the specific parsed parameters
        dataset_args = self._get_dataset_constructor_arguments(Dataset)
        dataset_arg_names = list(dataset_args.keys())
        dataset_constructor_args = {}
        # TODO: Add dataset specific arguments to be logged
        for k, v in self.args.items():
            if k in dataset_arg_names:
                dataset_constructor_args[k] = v
        # # Add dataset params
        dataset_constructor_args['data_name'] = self.args['data_name']
        dataset = self._init_dataset(Dataset, dataset_constructor_args)
        return dataset

    def _init_dataset(self, Dataset, dataset_args):
        # TODO: Override this if our model has special inputs
        print(' -- Dataset Parameters --')
        for k, v in dataset_args.items():
            print('\t{}: {}'.format(k, v))
        dataset = Dataset(**dataset_args)
        # transform dataset to remove all dataframes
        split_dataframe_tr = SplitDataFramesTr()
        dataset = transform_dataset(dataset, transforms=(split_dataframe_tr))
        return dataset

    def _get_train_val_data(self):
        train_size = int(len(self.dataset) * self.args['train_fraction'])
        val_size = len(self.dataset) - train_size
        train_data, val_data = random_split(self.dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(self.args['seed']))
        return train_data, val_data

    def _get_loaders(self):
        train_data, val_data = self._get_train_val_data()
        train_size = len(train_data)
        val_size = len(val_data)
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
        self.args['dataset_params'] = dataset_params
        self.args['input_sizes'] = sizes

        return train_loader, val_loader

    def _get_model(self):
        Model = self.models_dict[self.args['model_name']]  # Select the model class

        # Get the specific parsed parameters
        model_arg_names = list(self._get_model_constructor_arguments(Model))
        model_args = {}
        for k, v in self.args.items():
            if k in model_arg_names:
                model_args[k] = v
        # Add dataset params
        model_args['dataset_params'] = self.args['dataset_params']
        model_args['dataset_params'] = self.args['dataset_params']
        # TODO: Add dataset specific arguments to be logged
        model = self._init_model(Model, model_args)
        return model 
    
    def _init_model(self, Model, model_args):
        # TODO: Override this if our model has special inputs
        print(' -- Model Parameters --')
        for k, v in model_args.items():
            print('\t{}: {}'.format(k, v))
        model = Model(**model_args)
        print(' -- MODEL -- ')
        print(model)
        return model

    def _get_logger(self):
        logger = TensorBoardLogger(os.path.join(self.args['data_name'], 'tb_logs'), name=self.model.name)
        return logger

    def train(self, gpu=None):
        logger = self._get_logger()
        gpus = 0
        if gpu is None:
            gpu = not self.args['no_gpu']
        if torch.cuda.is_available() and gpu:
            gpus = 1
        trainer = pl.Trainer(gpus=gpus, max_epochs=self.args['max_epochs'], logger=logger)
        trainer.fit(self.model, self.train_loader, self.val_loader)

    def _get_dataset_constructor_arguments(self, Dataset):
        # It turns out that because Datasets inehrit from ABC, inspect does not have access to parents arguments.
        # Therefore, this hacks it by recursively explorign the partents
        constructor_arguments = {}
        exclude_args = ['self', 'args', 'kwargs']
        # Recursively search for all parameters:
        current_params = inspect.signature(Dataset.__init__).parameters
        for k,v in current_params.items():
            if k not in exclude_args:
                if v.kind is not v.empty:
                    constructor_arguments[k] = v.default
        # forward the parent classes:
        for parent_class in Dataset.__bases__:
            if issubclass(parent_class, DatasetBase):
                parent_params = self._get_dataset_constructor_arguments(parent_class)
                # combine
                for k,v in parent_params.items():
                    if k not in constructor_arguments:
                        constructor_arguments[k] = v
        return constructor_arguments

    def _get_model_constructor_arguments(self, Model):
        # It turns out that because Datasets inehrit from ABC, inspect does not have access to parents arguments.
        # Therefore, this hacks it by recursively explorign the partents
        constructor_arguments = {}
        exclude_args = ['self', 'args', 'kwargs']
        # Recursively search for all parameters:
        current_params = inspect.signature(Model.__init__).parameters
        for k, v in current_params.items():
            if k not in exclude_args:
                if v.kind is not v.empty:
                    constructor_arguments[k] = v.default
        # forward the parent classes:
        for parent_class in Model.__bases__:
            if issubclass(parent_class, pl.LightningModule):
                parent_params = self._get_model_constructor_arguments(parent_class)
                # combine
                for k, v in parent_params.items():
                    if k not in constructor_arguments:
                        constructor_arguments[k] = v
        return constructor_arguments

