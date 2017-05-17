from surprise.chart import *

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


# # dataset
# dataset_path1 = os.path.expanduser('./Dataset/ml-20m/')
# dataset_path2 = os.path.expanduser('./Dataset/LT/')

# # ml_dataset = Dataset(dataset_path=dataset_path1, tag_genome=False)
# lt_dataset = Dataset(dataset_path=dataset_path2,
#                      tag_genome=False, LT=True)
# lt_dataset.split(n_folds=5)
# lt_dataset.info()

import pickle

dump_dir = os.path.expanduser('~') + '/Thesis/experiment/dumps/usage_parts'

# grid_search_result1 = pickle.load(
#     open(os.path.join(dump_dir, 'grid_search_result/ItemRelTags-factors_10_200'), 'rb'))
# grid_search_result2 = pickle.load(
#     open(os.path.join(dump_dir, 'grid_search_result/ItemTopics-factors_10_200'), 'rb'))


# plt.plot([param['n_factors'] for param in grid_search_result1['predictions']['params']], grid_search_result1['predictions']['RMSE'], label='ItemRelTags')
# plt.plot([param['n_factors'] for param in grid_search_result2['predictions']['params']], grid_search_result2['predictions']['RMSE'], label='ItemTopics')
# # plt.title('compare')
# plt.legend()
# plt.show()


SVD = pickle.load(open(os.path.join(dump_dir, 'SVD-ml-unb-glo-5parts'), 'rb'))
ItemTopics = pickle.load(open(os.path.join(dump_dir, 'ItemTopics-ml-unb-glo-5parts'), 'rb'))
ItemRelTags = pickle.load(open(os.path.join(dump_dir, 'ItemRelTags-ml-unb-glo-5parts'), 'rb'))
CrossItemTopics = pickle.load(open(os.path.join(dump_dir, 'CrossItemTopics-ml-unb-glo-5parts'), 'rb'))


compare_part_usage_perf([SVD, ItemTopics, ItemRelTags, CrossItemTopics])






