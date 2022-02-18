import os
import torch
from bubble_utils.bubble_datasets.combined_dataset import CombinedDataset
from bubble_control.bubble_learning.datasets.bubble_drawing_dataset import BubbleDrawingDataset
from bubble_pivoting.datasets.bubble_pivoting_dataset import BubblePivotingDataset, BubblePivotingDownsampledDataset
from bubble_control.bubble_learning.datasets.dataset_wrappers import BubbleImprintCombinedDatasetWrapper
from bubble_control.bubble_learning.aux.orientation_trs import QuaternionToAxis
from bubble_control.bubble_learning.datasets.fixing_datasets.fix_object_pose_encoding_processed_data import EncodeObjectPoseAsAxisAngleTr
from bubble_utils.bubble_datasets.data_transformations import TensorTypeTr


class TaskCombinedDataset(CombinedDataset):

    def __init__(self, data_name, downsample_factor_x=7, downsample_factor_y=7, downsample_reduction='mean', **kwargs):
        self.data_dir = data_name # it assumes that all datasets are found at the same directory called data_dir
        self.downsample_factor_x = downsample_factor_x
        self.downsample_factor_y = downsample_factor_y
        self.downsample_reduction = downsample_reduction
        datasets = self._get_datasets()
        super().__init__(datasets, data_name=os.path.join(self.data_dir, 'task_combined_dataset'), **kwargs)

    @classmethod
    def get_name(self):
        return 'task_combined_dataset'

    def _get_datasets(self):
        datasets = []
        trs = [QuaternionToAxis(), EncodeObjectPoseAsAxisAngleTr(), TensorTypeTr(dtype=torch.float32)]
        drawing_dataset_line = BubbleDrawingDataset(
            data_name=os.path.join(self.data_dir, 'drawing_data_one_direction'),
            downsample_factor_x=self.downsample_factor_x,
            downsample_factor_y=self.downsample_factor_y,
            downsample_reduction=self.downsample_reduction,
            wrench_frame='med_base',
            transformation=trs,
        )
        datasets.append(drawing_dataset_line)
        drawing_dataset_one_dir = BubbleDrawingDataset(
            data_name=os.path.join(self.data_dir, 'drawing_data_line'),
            downsample_factor_x=self.downsample_factor_x,
            downsample_factor_y=self.downsample_factor_y,
            downsample_reduction=self.downsample_reduction,
            wrench_frame='med_base',
            transformation=trs,
        )
        datasets.append(drawing_dataset_one_dir)
        pivoting_dataset = BubblePivotingDownsampledDataset(
            data_name=os.path.join(self.data_dir, 'bubble_pivoting_data'),
            downsample_factor_x=self.downsample_factor_x,
            downsample_factor_y=self.downsample_factor_y,
            downsample_reduction=self.downsample_reduction,
            wrench_frame='med_base',
            transformation=trs,
        )
        datasets.append(pivoting_dataset)

        # Make them combined datasets:
        combined_datasets = [BubbleImprintCombinedDatasetWrapper(dataset) for dataset in datasets]
        return combined_datasets


if __name__ == '__main__':
    task_combined_dataset = TaskCombinedDataset('/home/mmint/bubble_datasets', only_keys=['imprint'])
    d0 = task_combined_dataset[0]
    print(d0)

