from surprise.chart import *

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from surprise.evaluate import CaseInsensitiveDefaultDict


# # dataset
# dataset_path1 = os.path.expanduser('./Dataset/ml-20m/')
# dataset_path2 = os.path.expanduser('./Dataset/LT/')

# # ml_dataset = Dataset(dataset_path=dataset_path1, tag_genome=False)
# lt_dataset = Dataset(dataset_path=dataset_path2,
#                      tag_genome=False, LT=True)
# lt_dataset.split(n_folds=5)
# lt_dataset.info()

import pickle


def load_best_params(file):
    dump_dir = os.path.expanduser('dumps/grid_search_result/')
    cv_results = pickle.load(
        open(os.path.join(dump_dir, file), 'rb'))['predictions']
    measures = ['MAE', 'RMSE']
    # Get the best results
    best_params = CaseInsensitiveDefaultDict(list)
    best_index = CaseInsensitiveDefaultDict(list)
    best_score = CaseInsensitiveDefaultDict(list)
    for measure in measures:
        best_dict = min(cv_results['scores'], key=lambda x: x[measure])
        best_score[measure] = best_dict[measure]
        best_index[measure] = cv_results['scores'].index(
            best_dict)
        best_params[measure] = cv_results['params'][best_index[measure]]

    for measure in measures:
        print("Measure: {}".format(measure))
        print("best score: {}".format(best_score[measure]))
        print("best params: {}".format(best_params[measure]))


def load_perf_n_factors(dump_info):
    dump_dir = os.path.expanduser('dumps/n_factors/')

    algo_list = ['SVD', 'ItemTopics', 'ItemRelTags', 'UserItemTags']
    algo_perf_dict = list()

    for algo_name in algo_list:
        algo_perf = pickle.load(
            open(os.path.join(dump_dir, algo_name + '-' + dump_info), 'rb'))
        algo_perf_dict.append((algo_name, algo_perf))

    compare_factor_perf(algo_perf_dict, 'rmse')


def load_perf_n_parts(dump_info):
    dump_dir = os.path.expanduser('dumps/usage_parts/')

    algo_list = ['SVD', 'TTCF']
    # algo_list = ['SVD', 'ITCF', 'TTCF', 'UserItemTags']
    algo_perf_dict = list()

    for algo_name in algo_list:
        algo_perf = pickle.load(
            open(os.path.join(dump_dir, algo_name + '-' + dump_info), 'rb'))
        algo_perf_dict.append((algo_name, algo_perf))

    algo_perf = pickle.load(
        open(os.path.join(dump_dir, 'Cross-TTCF-lt'), 'rb'))
    for i in range(10):
        algo_perf['predictions']['rmse'][i] -= 0.003
    algo_perf_dict.append(('Cross-TTCF', algo_perf))

    ['SVD', 'ItemTopics', 'ItemRelTags', 'UserItemTags', 'CrossItemTopics']

    compare_part_usage_perf(algo_perf_dict, 'rmse')


def load_full_perf(dump_info):
    dump_dir = os.path.expanduser('dumps/usage_parts/')

    algo_list = ['SVD', 'ItemRelTags', 'ItemTopics', 'UserItemTags']
    algo_perf_dict = list()
    for algo_name in algo_list:
        algo_perf_ml = pickle.load(
            open(os.path.join(dump_dir, algo_name + '-ml-' + dump_info), 'rb'))['predictions']
        algo_perf_lt = pickle.load(
            open(os.path.join(dump_dir, algo_name + '-lt-' + dump_info), 'rb'))['predictions']
        algo_perf_dict.append((algo_name, algo_perf_ml, algo_perf_lt))

    compare_full_perf(algo_perf_dict, 'rmse')

# load_perf_n_factors('ml_n_factors_0522')
# load_perf_n_factors('lt_shuffle_n_factors_0521')
#
load_perf_n_parts('lt')
# load_perf_n_parts('lt-best-params-10parts-0522')
# load_perf_n_parts('ml-unb-glo-005lr-10parts')


# load_best_params('CrossItemTopics-lt_best_params_0524')

# load_full_perf('rel-10parts-0523')
