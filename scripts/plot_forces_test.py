#! /usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_forces():
    path_to_data = '~/Desktop/med_wrench_recording.csv'
    df = pd.read_csv(path_to_data)
    labels = ['wrench.force.x','wrench.force.y','wrench.force.z']
    fig = plt.figure()
    for label in labels:
        plt.plot(df[label],label=label)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_forces()
