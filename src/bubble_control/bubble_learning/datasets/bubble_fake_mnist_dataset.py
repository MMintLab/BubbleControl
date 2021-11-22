import numpy as np
import tf.transformations as tr

from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase



class BubbleFakeMNISTDataset(BubbleDatasetBase):

    def __init__(self, *args, train=True, **kwargs):
        self.train = train
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        dataset_type = 'train'
        if not self.train
            dataset_type = 'test'
        return 'bubble_fake_mnist_dataset_{}'.format(dataset_type)

    def _load

    def _get_sample(self):
    