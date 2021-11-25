import numpy as np
import tf.transformations as tr
import torchvision

from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase


class BubbleFakeMNISTDataset(BubbleDatasetBase):

    def __init__(self, *args, train=True, **kwargs):
        self.train = train
        super().__init__(*args, **kwargs)
        self.mnist_dataset = self._get_mnist_dataset()

    @classmethod
    def get_name(cls):
        return 'bubble_fake_mnist_dataset'

    @property
    def name(self):
        dataset_type = 'train'
        if not self.train:
            dataset_type = 'test'
        return '{}_{}'.format(self.get_name(), dataset_type)

    def _get_mnist_dataset(self):
        import pdb; pdb.set_trace()
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

    def __len__(self):
        return self.mnist_dataset.__len__()

    def _get_sample(self, idx):
        return self.mnist_dataset[idx]

