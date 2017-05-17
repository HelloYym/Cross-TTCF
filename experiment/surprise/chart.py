import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


def part_usage_perf(parts_performances):
    ''' 采用不同比例训练集的性能

    '''

    # usage = [format(f, '.0%') for f in np.arange(0.1, 1.1, 0.1)]
    usage = np.arange(0.2, 1.1, 0.2)
    plt.plot(usage, parts_performances['mae'], label='RMSE')

    plt.legend()
    plt.show()


def compare_part_usage_perf(n_parts_perf, measure='rmse'):
    ''' 采用不同比例训练集的性能

    '''

    # usage = [format(f, '.0%') for f in np.arange(0.1, 1.1, 0.1)]
    usage = np.arange(0.1, 1.1, 0.1)

    marker_list = ['.', 'o', 'x', 'v', '*', 'p', 's']

    for i, perf in enumerate(n_parts_perf):
        plt.plot(usage, perf['predictions'][measure], label=i, marker=marker_list[i])

    plt.title('compare')
    plt.legend()
    plt.show()


def compare_factor_perf(n_parts_perf, measure='rmse'):
    ''' 采用不同比例训练集的性能

    '''

    # usage = [format(f, '.0%') for f in np.arange(0.1, 1.1, 0.1)]
    usage = np.arange(0.1, 1.1, 0.1)
    measure = measure.upper()
    marker_list = ['.', 'o', 'x', '+', '*']
    for i, pred in enumerate(n_parts_perf):
        plt.plot(usage, [perf[measure] for perf in pred['predictions']['scores']], label=i, marker=marker_list[i])

    plt.title('compare')
    plt.legend()
    plt.show()



