import numpy as np
from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase


class BubbleImprintCombinedDatasetWrapper(BubbleDatasetBase):
    """
    Creates a new dataset from the original wrapped dataset which is the the dataset in 2 so samples contain 'imprint' which is the combination of 'init_imprint' and 'final_imprint'
    """
    def __init__(self, dataset):
        self.dataset = dataset
        super().__init__(data_name=dataset.data_path)

    def _get_filecodes(self):
        filecodes = np.arange(2*len(self.dataset))
        return filecodes

    def _get_sample(self, fc):
        true_fc = fc // 2
        sample = self.dataset[true_fc]
        if fc % 2 == 0:
            # sample is the initial
            sample['imprint'] = sample['init_imprint']
        else:
            sample['imprint'] = sample['final_imprint']
        return sample

    def get_name(self):
        name = '{}_imprint_combined'.format(self.dataset.name)
        return name
