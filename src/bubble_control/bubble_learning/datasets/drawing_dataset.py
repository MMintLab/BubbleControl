import os

from bubble_utils.bubble_datasets.combined_dataset import CombinedDataset
from bubble_control.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from bubble_control.bubble_learning.datasets.dataset_wrappers import BubbleImprintCombinedDatasetWrapper


class DrawingDataset(CombinedDataset):

    def __init__(self, data_dir, downsample_factor_x=7, downsample_factor_y=7, downsample_reduction='mean', **kwargs):
        self.data_dir = data_dir # it assumes that all datasets are found at the same directory called data_dir
        self.downsample_factor_x = downsample_factor_x
        self.downsample_factor_y = downsample_factor_y
        self.downsample_reduction = downsample_reduction
        datasets = self._get_datasets()
        super().__init__(datasets, data_name=os.path.join(self.data_dir, 'task_combined_dataset'), **kwargs)

    @classmethod
    def get_name(self):
        return 'drawing_dataset'

    def _get_datasets(self):
        datasets = []
        drawing_dataset_line = BubbleDrawingDataset(
            data_name=os.path.join(self.data_dir, 'drawing_data_one_direction'),
            downsample_factor_x=self.downsample_factor_x,
            downsample_factor_y=self.downsample_factor_y,
            downsample_reduction=self.downsample_reduction)
        datasets.append(drawing_dataset_line)
        drawing_dataset_one_dir = BubbleDrawingDataset(
            data_name=os.path.join(self.data_dir, 'drawing_data_line'),
            downsample_factor_x=self.downsample_factor_x,
            downsample_factor_y=self.downsample_factor_y,
            downsample_reduction=self.downsample_reduction)
        datasets.append(drawing_dataset_one_dir)

        # Make them combined datasets:
        return datasets

if __name__ == '__main__':
    drawing_combined_dataset = DrawingDataset('/home/mik/Datasets/bubble_datasets')
    d0 = drawing_combined_dataset[0]
    print(d0)