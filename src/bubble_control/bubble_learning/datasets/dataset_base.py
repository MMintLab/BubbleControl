import numpy as np
import sys
import os
import pandas as pd
from torch.utils.data import Dataset
import abc
import matplotlib.pyplot as plt
import torch
import shutil
from tqdm import tqdm

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_control')[0], 'bubble_control')


class DatasetBase(Dataset, abc.ABC):

    def __init__(self, data_name, transformation=None, dtype=torch.float):
        self.dtype = dtype
        self.data_path, self.data_name = self.__get_data_path(data_name)
        self.processed_data_path = self._get_processed_data_path()
        self.dl = pd.read_csv(os.path.join(self.data_path, '{}_DataLegend.csv'.format(self.data_name)))
        self.fcs = self._get_filecodes()
        self.transformation = transformation

        super().__init__()
        self.means, self.stds = None, None
        self.maxs, self.mins = None, None
        self.process()

    # ========================================================================================================
    #       IMPLEMENT/OVERWRITE FUNCTIONS:
    # ---------------------------

    @abc.abstractmethod
    def _get_sample(self, fc):
        """
        Retruns the sample corresponding to the filecode fc
        :param fc:
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def name(self):
        """
        Returns an unique identifier of the dataset
        :return:
        """
        pass

    def _get_filecodes(self):
        """
        Return a list containing the data filecodes.
        Overwrite the function in case the data needs to be filtered.
        By default we load all the filecodes we find in the datalegend
        :return:
        """
        return self.dl['FileCode'].to_numpy()

    # ========================================================================================================
    #       BASIC FUNCTIONS:
    # ---------------------------

    def __len__(self):
        dataset_len = len(self.fcs)
        return dataset_len

    def _get_processed_data_path(self):
        return os.path.join(self.data_path, 'processed_data', self.name)

    def _get_project_path(self):
        """
        Returns the path to the main project directory. Used for finding the default data path
        :return:
        """
        return project_path

    def _get_processed_sample(self, item):
        # each data sample is composed by:
        #   'mask': object contact mask
        #   'y': resulting delta motion
        fc = self.fcs[item]
        sample = self._get_sample(fc)
        if self.transformation is not None:
            if type(self.transformation) is list:
                for tr in self.transformation:
                    sample = tr(sample)
            else:
                sample = self.transformation(sample)
        return sample

    def __getitem__(self, item):
        save_path_i = os.path.join(self.processed_data_path, 'data_{}.pt'.format(item))
        item_i = torch.load(save_path_i)
        return item_i

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def process(self):
        if not os.path.exists(self.processed_data_path):
            print('Processing the data and saving it to {}'.format(self.processed_data_path))
            os.makedirs(self.processed_data_path)
            # process the data and save it as .pt files
            try:
                for i in tqdm(range(self.__len__())):
                    sample_i = self._get_processed_sample(i)
                    sample_i = self._tr_sample(sample_i)
                    save_path_i = os.path.join(self.processed_data_path, 'data_{}.pt'.format(i))
                    torch.save(sample_i, save_path_i)
            except:
                print('Removing the directory since the processing was stopped due to an error.')
                # remove the data directory since some error has occurred
                shutil.rmtree(self.processed_data_path)
                raise
            print('Data processed')

    def _tr_sample(self, sample_i):
        for k, v in sample_i.items():
            sample_i[k] = torch.tensor(v, dtype=self.dtype)
        return sample_i

    def get_sizes(self):
        sample_test = self.__getitem__(0)
        sizes = {}
        for item, value in sample_test.items():
            size_i = value.shape
            # sizes[item] = value.shape[-1]  # Old way, only last value
            if len(size_i) == 1:
                sizes[item] = value.shape[-1]
            else:
                sizes[item] = np.asarray(value.shape)
        return sizes

    def invert(self, sample):
        # apply the inverse transformations
        if self.transformation is not None:
            if type(self.transformation) is list:
                for tr in reversed(self.transformation):
                    sample = tr.inverse(sample)
            else:
                sample = self.transformation.inverse(sample)
        return sample

    # ========================================================================================================
    #       AUXILIARY FUNCTIONS:
    # ---------------------------

    def __get_data_path(self, data_name):
        data_path = None
        real_data_name = None
        if data_name.startswith("/"):
            # we provide the absolute path
            data_path = data_name
            real_data_name = data_path.split('/')[-1]
        elif '/' in data_name:
            # we provide the relative path
            data_path = os.path.join(os.getcwd(), data_name)
            real_data_name = data_path.split('/')[-1]
        else:
            # we just provide the data name
            project_path = self._get_project_path()
            data_path = os.path.join(project_path, 'data', data_name)
            real_data_name = data_name
        return data_path, real_data_name

    def _process_list_array(self, list_array_raw):
        _list_array_raw = list_array_raw[1:-1] # remove list "'" and "'"
        list_elem = _list_array_raw.split(' ')
        list_out = [float(x) for x in list_elem if x!='']
        return np.array(list_out)

    def _process_str_list(self, str_list):
        _str_list = str_list[1:-1] #remove "[" and "]"
        list_elem = _str_list.split(', ')
        list_out = [float(x) for x in list_elem if x!='']
        return np.array(list_out)

    def _process_str_list_of_str(self, str_list):
        _str_list = str_list[1:-1]  # remove "[" and "]"
        list_elem = _str_list.split(', ')
        list_out = [x[1:-1] for x in list_elem if x != ''] # remove the '' around each string
        return list_out

    def _pack_all_samples(self):
        sample_test = self.__getitem__(0)
        samples = {}
        for key in sample_test.keys():
            samples[key] = []
        for i in range(self.__len__()):
            sample_i = self.__getitem__(i)
            for key in samples.keys():
                samples[key].append(sample_i[key].detach().cpu().numpy())
        for key,value in samples.items():
            samples[key] = np.stack(value, axis=0)
        return samples

    # ========================================================================================================
    #       NORMALIZATION AND STANDARDIZATION FUNCTIONS:
    # ---------------------------------------------------

    def _get_storage_path(self):
        # Override this function in case we want to modify the path
        return os.path.join(self.data_path, 'stand_const', self.name)

    def save_standardization_constants(self):
        save_path = self._get_storage_path()
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'means.npy'), self.means)
        np.save(os.path.join(save_path, 'stds.npy'), self.stds)
        np.save(os.path.join(save_path, 'maxs.npy'), self.maxs)
        np.save(os.path.join(save_path, 'mins.npy'), self.mins)

    def load_standardization_constants(self):
        load_path = self._get_storage_path()
        if not os.path.isfile(os.path.join(load_path, 'means.npy')):
            # there are no normalization constants saved, so compute them
            self._compute_standardization_constants()
        self.means = np.load(os.path.join(load_path, 'means.npy')).item()
        self.stds = np.load(os.path.join(load_path, 'stds.npy')).item()
        self.maxs = np.load(os.path.join(load_path, 'maxs.npy')).item()
        self.mins = np.load(os.path.join(load_path, 'mins.npy')).item()
        print('Normalization constants loaded.')
        return self.means, self.stds

    def _compute_standardization_constants(self):
        sample_test = self.__getitem__(0)
        means = {}
        stds = {}
        maxs = {}
        mins = {}
        print('Computing the dataset standardization constants, please wait')

        samples = self._pack_all_samples()
        for key in samples.keys():
            samples_i = samples[key]
            mean_i = np.mean(samples_i, axis=0)
            std_i = np.std(samples_i, axis=0)
            max_i = np.max(samples_i, axis=0)
            min_i = np.min(samples_i, axis=0)
            # Remove noise from std. If it is less than 5 orders of magnitude than the mean, set it to 0
            _mean_std_orders_magnitude = np.abs(np.divide(mean_i, std_i, out=np.zeros_like(mean_i), where=std_i != 0))
            mean_std_orders_magnitude = np.log10(_mean_std_orders_magnitude,
                                                 out=np.zeros_like(_mean_std_orders_magnitude),
                                                 where=_mean_std_orders_magnitude != 0)
            std_i[np.where(mean_std_orders_magnitude > 5)] = 0

            means[key] = mean_i
            stds[key] = std_i
            maxs[key] = max_i
            mins[key] = min_i
        print('\t -- Done!')
        self.means = means
        self.stds = stds
        self.maxs = maxs
        self.mins = mins
        self.save_standardization_constants()
        return means, stds

    # ========================================================================================================
    #       DATA DEBUGGING FUNCTIONS:
    # ----------------------------------

    def get_histograms(self, num_bins=100):
        histogram_path = os.path.join(self.data_path, 'data_stats', 'histograms', self.name)
        print('\n -- Getting Histograms --')
        print('Packing all samples, please wait...')
        samples = self._pack_all_samples()
        print('Computing histograms')
        for key, sample_i in samples.items():
            hist_key_path = os.path.join(histogram_path, key)
            if not os.path.isdir(hist_key_path):
                os.makedirs(hist_key_path)
            num_features = sample_i.shape[-1]
            for feat_indx in range(num_features):
                data_i = sample_i[:, feat_indx]
                fig = plt.figure()
                plt.hist(data_i, color='blue', edgecolor='black',
                     bins=num_bins)
                plt.title('Dataset {} - {} feature {}'.format(self.name, key, feat_indx))
                plt.xlabel('{} feature {}'.format(key, feat_indx))
                plt.ylabel('Counts')
                plt.savefig(os.path.join(hist_key_path, '{}_feature_{}_hist.png'.format(key, feat_indx)))
                plt.close()
            # Save also the statistics:
            mean_i = np.mean(sample_i, axis=0)
            std_i = np.std(sample_i, axis=0)
            max_i = np.max(sample_i, axis=0)
            min_i = np.min(sample_i, axis=0)
            q_1 = np.quantile(sample_i, q=0.25, axis=0)
            q_2 = np.quantile(sample_i, q=0.5,  axis=0)
            q_3 = np.quantile(sample_i, q=0.75, axis=0)

            _mean_std_orders_magnitude = np.abs(np.divide(mean_i, std_i, out=np.zeros_like(mean_i), where=std_i != 0))
            mean_std_orders_magnitude = np.log10(_mean_std_orders_magnitude,
                                                 out=np.zeros_like(_mean_std_orders_magnitude),
                                                 where=_mean_std_orders_magnitude != 0)
            std_i[np.where(mean_std_orders_magnitude > 5)] = 0

            feature_indxs = np.arange(num_features,dtype=np.int)
            stats_data = np.stack([feature_indxs, mean_i, std_i, max_i, min_i, q_1, q_2, q_3]).T

            df = pd.DataFrame(data=stats_data, columns=['Feat Indx', 'Mean', 'Std', 'Max', 'Min', 'Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)'])
            df.to_csv(os.path.join(hist_key_path, '{}_stats.csv'.format(key)), index=False)
        print('Histograms saved!')

    def get_scatterplots(self, num_samples=250):
        scatterplot_path = os.path.join(self.data_path, 'data_stats', 'scatter_plots', self.name)
        print('\n -- Getting Scatterplots --')
        print('Packing all samples, please wait...')
        samples = self._pack_all_samples()
        print('Computing scatterplots')
        for key_i, sample_i in samples.items():
            hist_key_path = os.path.join(scatterplot_path, key_i)
            if not os.path.isdir(hist_key_path):
                os.makedirs(hist_key_path)
            num_features_i = sample_i.shape[-1]
            for feat_indx_i in range(num_features_i):
                data_i = sample_i[:num_samples, feat_indx_i]
                for key_j, sample_j in samples.items():
                    num_features_j = sample_j.shape[-1]
                    for feat_indx_j in range(num_features_j):
                        data_j = sample_j[:num_samples, feat_indx_j]
                        fig = plt.figure()
                        plt.scatter(data_i, data_j, marker='.', color='blue')
                        plt.title('Dataset {} - {} feature {} vs {} feature {}'.format(self.name, key_i, feat_indx_i, key_j, feat_indx_j))
                        plt.xlabel('{} feature {}'.format(key_i, feat_indx_i))
                        plt.ylabel('{} feature {}'.format(key_j, feat_indx_j))
                        plt.savefig(os.path.join(hist_key_path, '{}_feature_{}_vs_{}_feature_{}_scatter.png'.format(key_i, feat_indx_i, key_j, feat_indx_j)))
                        plt.close()

        print('Scatterplots saved!')

    def detect_outliers(self):
        """
        Finds values that deviate to much from the norm
        :return:
        """
        # We may need to change the name of the this function
        outliers = {} # this will hold all the file codes that result outside the range
        sample_test = self.__getitem__(0)
        keys = sample_test.keys()
        if self.means is None:
            self._compute_standardization_constants()
        upp_vals, low_vals = {}, {}
        for key in keys:
            upp_vals[key] = self.means[key] + 4 * self.stds[key]
            low_vals[key] = self.means[key] - 4 * self.stds[key]

        for i in range(self.__len__()):
            sample_i = self.__getitem__(i)
            fc = self.fcs[i]
            for key in keys:
                out_range = np.logical_or(sample_i[key] >= upp_vals[key], sample_i[key]<=low_vals[key])
                if out_range.any():
                    # Mark it as outlier
                    if fc in outliers:
                        outliers[fc].append((key, list(np.where(out_range)[0])))
                    else:
                        outliers[fc] = [(key, list(np.where(out_range)[0]))]
        return outliers