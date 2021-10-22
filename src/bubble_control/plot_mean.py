#! /usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import os
from tf import transformations as tr
import pdb

class Plotter:
    def __init__(self, filename, title):
        self.filename = filename
        self.title = title

    def df_concat(self, directory):
        df = pd.DataFrame([])
        for file in os.listdir(directory):
            if file.endswith(".csv"):                
                new_df = pd.read_csv(os.path.join(directory, file))
                if (('header.stamp.secs' in new_df) & ('header.stamp.nsecs' in new_df)):
                    new_df['Time'] = new_df['header.stamp.secs'] + new_df['header.stamp.nsecs']*10**(-9)
                if 'Time' in new_df:
                    new_df['Time'] -= new_df['Time'][0]
                df = pd.concat([df, new_df])
        return df

    def plot_mean(self, directory, magnitudes):
        df = self.df_concat(directory)
        df_mean = df.groupby(level=0).mean()
        figs = []
        for magnitude, labels in magnitudes.items():
            figs.append(plt.figure())
            for label in labels:
                plt.plot(df_mean['Time'], df_mean[label],label=label)
            plt.legend()
            plt.title(self.title + magnitude)
            plt.xlabel('Time (s)')
            plots_directory = '{}../plots/'.format(directory)
            if not os.path.isdir(plots_directory):
                print('creating directory')
                os.makedirs(plots_directory)
            plt.savefig('{}/{}_{}.png'.format(plots_directory, self.filename, magnitude))

    def plot_mean_std(self, directory, magnitudes):
        df = self.df_concat(directory)
        df_mean = df.groupby(level=0).mean()
        df_std = df.groupby(level=0).std()
        figs = []
        for magnitude, labels in magnitudes.items():
            viridis = cm.get_cmap('viridis', 256)
            colors = viridis(np.linspace(0, 1, len(labels)))
            figs.append(plt.figure())
            for i, label in enumerate(labels):
                plt.plot(df_mean['Time'], df_mean[label],label=label, color=colors[i])
                plt.plot(df_mean['Time'], df_mean[label] - df_std[label], color=colors[i], alpha=0.5)
                plt.plot(df_mean['Time'], df_mean[label] + df_std[label], color=colors[i], alpha=0.5)
            plt.legend()
            plt.title(self.title + magnitude)
            plt.xlabel('Time (s)')
            plots_directory = '{}../plots/'.format(directory)
            if not os.path.isdir(plots_directory):
                print('creating directory')
                os.makedirs(plots_directory)
            plt.savefig('{}/{}_{}.png'.format(plots_directory, self.filename, magnitude))

    def plot_angles_mean(self, directory, axes):
        df = self.df_concat(directory)
        quat = df.iloc[:,5:9]
        euler = quat.apply(tr.euler_from_quaternion, axis=1, result_type='expand')
        df['euler_x'] = euler.iloc[:,0]
        df['euler_y'] = euler.iloc[:,1]
        df['euler_z'] = euler.iloc[:,2]
        df_mean = df.groupby(level=0).mean()
        plt.figure()
        for axis in axes:
            plt.plot(df_mean['Time'], df_mean['euler_'+axis],label=axis)
        plt.title(self.title + 'rotation trajectory')
        plt.xlabel('Time (s)')
        plt.ylabel('Tool\'s rotation angle (rad)')
        plots_directory = '{}../plots/'.format(directory)
        if not os.path.isdir(plots_directory):
            print('creating directory')
            os.makedirs(plots_directory)
        plt.savefig('{}/{}_{}.png'.format(plots_directory, self.filename, 'rotation_trajectory'))

    def plot_angles_mean_std(self, directory, axes):
        df = self.df_concat(directory)
        quat = df.iloc[:,5:9]
        euler = quat.apply(tr.euler_from_quaternion, axis=1, result_type='expand')
        df['euler_x'] = euler.iloc[:,0]
        df['euler_y'] = euler.iloc[:,1]
        df['euler_z'] = euler.iloc[:,2]
        df_mean = df.groupby(level=0).mean()
        df_std = df.groupby(level=0).std()
        viridis = cm.get_cmap('viridis', 256)
        colors = viridis(np.linspace(0, 1, len(axes)))
        plt.figure()
        for i, axis in enumerate(axes):
            plt.plot(df_mean['Time'], df_mean['euler_'+axis],label=axis, color=colors[i])
            plt.plot(df_mean['Time'], df_mean['euler_'+axis] - df_std['euler_'+axis], color=colors[i], alpha=0.5)
            plt.plot(df_mean['Time'], df_mean['euler_'+axis] + df_std['euler_'+axis], color=colors[i], alpha=0.5)
        plt.title(self.title + 'rotation trajectory')
        plt.xlabel('Time (s)')
        plt.ylabel('Tool\'s rotation angle (rad)')
        plots_directory = '{}../plots/'.format(directory)
        if not os.path.isdir(plots_directory):
            print('creating directory')
            os.makedirs(plots_directory)
        plt.savefig('{}/{}_{}.png'.format(plots_directory, self.filename, 'rotation_trajectory'))


if __name__ == '__main__':
    directory = '/home/mireiaplanaslisbona/Documents/experiments_data/repeated_experiments/putting_down/orientation2/putting_down_spatula_reversed'
    filename = 'putting_down_spatula_reversed'
    wrench_magnitudes = {'torque': ['wrench.torque.x','wrench.torque.y','wrench.torque.z'], 'force': ['wrench.force.x','wrench.force.y','wrench.force.z']}
    title = 'Putting down spatula reversed '
    plotter = Plotter(filename, title)
    for root, subdirectories, files in os.walk(directory, topdown=True):
        depth = root[len(directory) + len(os.path.sep):].count(os.path.sep)
        if ((depth == 0) & (root !=directory)):
            print(root)
            plotter.plot_mean_std('{}/wrench/'.format(root), wrench_magnitudes)
            plotter.plot_angles_mean_std('{}/tf/'.format(root), ['x'])



