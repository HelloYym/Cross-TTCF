import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict


def tag_power_law(tag_freq):
    ''' 统计标签的长尾分布

    '''
    plt.style.use('ggplot')
    fk = defaultdict(int)
    for freq in tag_freq.values():
        fk[freq] += 1

    x = list(fk.keys())
    y = np.array(list(fk.values()))

    # plt.scatter(x, y, c=np.random.rand(len(fk)))
    plt.scatter(x, y, s=50, c='#e76278')
    # plt.title('标签流行度的长尾分布')
    plt.xlabel('流行度')  # 给 x 轴添加标签
    plt.ylabel('标签频度')  # 给 y 轴添加标签
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(0)
    plt.ylim(0)
    plt.show()


def part_usage_perf(parts_performances):
    ''' 采用不同比例训练集的性能

    '''

    # usage = [format(f, '.0%') for f in np.arange(0.1, 1.1, 0.1)]
    usage = np.arange(0.2, 1.1, 0.2)
    plt.plot(usage, parts_performances['mae'], label='RMSE')

    plt.legend()
    plt.show()


def compare_part_usage_perf(algo_perf, measure='rmse'):
    ''' 采用不同比例训练集的性能

    '''
    plt.style.use('ggplot')

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    # line_styles = ['--', '-', '-', '--', '-x']
    # colors = ['#4ed5c7', '#e76278', '#52696f', '#b69968', '#403874']
    line_styles = ['--', '--', '-', '-', '-x']
    colors = ['#4ed5c7', '#b69968', '#e76278', '#52696f', '#403874']
    usage = np.arange(0.1, 1.1, 0.1)

    for i, (algo, perf) in enumerate(algo_perf):
        ax.plot(usage, perf['predictions'][measure],
                line_styles[i], linewidth=1.5, color=colors[i], label=algo)

    ax.set_xticklabels([format(f, '.0%') for f in np.arange(0.1, 1.1, 0.1)])
    plt.xlabel('训练数据用量')  # 给 x 轴添加标签
    plt.ylabel('RMSE')  # 给 y 轴添加标签
    plt.legend()
    plt.show()


def compare_full_perf(algo_perf, measure='rmse'):
    ''' 采用不同比例训练集的性能

    '''
    plt.style.use('ggplot')
    # plt.style.use('seaborn-white')
    colors = ['#4ed5c7', '#52696f', '#e76278', '#b8da8d']

    N = 2
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2       # the width of the bars
    fig, ax = plt.subplots()

    for i, (algo_name, perf1, perf2) in enumerate(algo_perf):
        ax.bar(ind + 0.1 + i * width, [perf1[measure][-1],
                                       perf2[measure][-1]], width=width, color=colors[i], label=algo_name)
    # add some text for labels, title and axes ticks
    ax.set_ylabel(measure.upper())
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(ind + 0.5)
    ax.set_xticklabels(('MovieLens', 'LibraryThings'))

    # plt.xlim(-1)
    plt.ylim(0.6)
    ax.legend()

    plt.show()


def compare_factor_perf(n_parts_perf, measure='rmse'):
    ''' 采用不同比例训练集的性能

    '''

    plt.style.use('ggplot')
    line_styles = ['r-o', 'c-s', 'm-^', 'y-*']

    usage = np.arange(10, 110, 10)
    measure = measure.upper()
    marker_list = ['.', 'o', 'x', '+', '*']
    colors = ['#4ed5c7', '#52696f', '#e76278', '#b8da8d']

    for i, (algo, pred) in enumerate(n_parts_perf):
        plt.plot(usage, [perf[measure]
                         for perf in pred['predictions']['scores']], label=algo, color=colors[i])

    # plt.title('compare')
    plt.xlabel('#factors')  # 给 x 轴添加标签
    plt.ylabel('RMSE')  # 给 y 轴添加标签
    plt.legend()
    plt.show()
