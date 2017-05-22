import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict


def tag_power_law(tag_freq):
    ''' 统计标签的长尾分布

    '''
    fk = defaultdict(int)
    for freq in tag_freq.values():
        fk[freq] += 1

    plt.scatter(list(fk.keys()), np.array(
        list(fk.values())), c=np.random.rand(len(fk)))
    plt.title('标签流行度的长尾分布')
    plt.xlabel('流行度')  # 给 x 轴添加标签
    plt.ylabel('标签频度')  # 给 y 轴添加标签
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(0)
    plt.ylim(-1)
    plt.show()


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
        plt.plot(usage, perf['predictions'][measure],
                 label=i, marker=marker_list[i])

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
        plt.plot(usage, [perf[measure]
                         for perf in pred['predictions']['scores']], label=i)

    plt.title('compare')
    plt.legend()
    plt.show()
