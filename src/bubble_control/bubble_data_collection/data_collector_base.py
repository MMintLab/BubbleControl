import numpy as np
import os
import sys
import time
import pandas as pd
import csv
import pickle
import abc
from collections import OrderedDict
from tqdm import tqdm

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_control')[0], 'bubble_control')
package_path = project_path
sys.path.append(project_path)


class DataCollectorBase(abc.ABC):

    def __init__(self, data_path=None):
        self.data_path, self.filename = self._get_data_path(data_path)
        self.datalegend_path = os.path.join(self.data_path, '{}_DataLegend.csv'.format(self.filename))
        self.last_filecode_pickle_path = os.path.join(self.data_path, 'lastFileCode.pickle')
        self.filecode = 1
        self._load()

    def _load(self):

        # Load or Create the data infrastructure if it does not exist
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        if not os.path.isfile(self.datalegend_path):
            legend_df = pd.DataFrame(columns=self._get_legend_column_names())
            legend_df.to_csv(self.datalegend_path, index=False)
            with open(self.last_filecode_pickle_path, 'wb') as f:
                pickle.dump(1, f)
        else:
            with open(self.last_filecode_pickle_path, 'rb') as f:
                self.filecode = pickle.load(f)

    @abc.abstractmethod
    def _get_legend_column_names(self):
        """
        Return a list containing the column names of the datalegend
        Returns:
        """
        pass

    @abc.abstractmethod
    def _get_legend_lines(self, data_params):
        """
        Return a list containing the values to log inot the data legend for the data sample with file code filecode
        Args:
            data_params: <dict> containg parameters of the collected data
        Returns:
        """
        pass

    @abc.abstractmethod
    def _collect_data_sample(self, params=None):
        """
        Collect and save data to the designed path in self.data_path
        Args:
            params:
        Returns: <dict> containing the parameters of the collected sample
        """
        pass

    def _get_data_path(self, data_path=None):
        if data_path is None:
            # Get some default directory based on the current working directory
            data_path = os.path.join(package_path, 'calibration_data')
        else:
            if data_path.startswith("/"):
                data_path = data_path  # we provide the full path (absolute)
            else:
                exec_path = os.getcwd()
                data_path = os.path.join(exec_path, data_path)  # we save the data on the path specified (relative)
        filename = data_path.split('/')[-1]
        return data_path, filename

    def collect_data(self, num_data):
        # Display basic information
        print('_____________________________')
        print(' Data collection has started!')
        print('  - The data will be saved at {}'.format(self.data_path))

        # Collect data
        pbar = tqdm(range(num_data), desc='Data Collected: ')
        num_data_collected = 1
        for i in pbar:
            pbar.set_postfix({'Filecode': self.filecode})
            # Save data
            sample_params = self._collect_data_sample()
            # Log data sample info to data legend
            legend_lines_vals = self._get_legend_lines(sample_params)
            num_data_collected = len(legend_lines_vals)
            self.filecode += num_data_collected # update the filecode
            with open(self.datalegend_path, 'a+') as csv_file:
                csv_file_writer = csv.writer(csv_file)
                for line_val in legend_lines_vals:
                    csv_file_writer.writerow(line_val)
            csv_file.close() # make sure it is closed

            # Update the filecode
            with open(self.last_filecode_pickle_path, 'wb') as f:
                pickle.dump(self.filecode, f)

