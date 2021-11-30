import numpy as np
import tf.transformations as tr
import torchvision

from bubble_utils.bubble_datasets.dataset_base import DatasetBase


class FakeMNISTDataset(DatasetBase):

    def __init__(self, *args, train=True, **kwargs):
        self.train = train
        self.mnist_dataset = None
        super().__init__(*args, **kwargs)

    def _get_datalegend(self):
        return None

    def _get_filecodes(self):
        self.mnist_dataset = self._get_mnist_dataset()
        filecodes = np.arange(len(self.mnist_dataset))
        return filecodes

    @classmethod
    def get_name(cls):
        return 'fake_mnist_dataset'

    @property
    def name(self):
        dataset_type = 'train'
        if not self.train:
            dataset_type = 'test'
        return '{}_{}'.format(self.get_name(), dataset_type)

    def _get_mnist_dataset(self):
        dataset = torchvision.datasets.MNIST(
            self.data_path,
            train=self.train,
            download=True, # TODO: Fix
            transform=torchvision.transforms.Compose([
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
           )
        return dataset

    def _get_sample(self, idx):
        mnist_sample = self.mnist_dataset[idx]
        sample = {
            'x': mnist_sample[0],
            'label': mnist_sample[1],                               
            }
        return sample
