
from surprise import Dataset
from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags, CrossUserItemRelTags, CrossItemRelTags, CrossItemTopics
from surprise import ItemTopicsTest
from surprise import GridSearch
from surprise.chart import *

import os
import numpy as np
import pandas as pd
import pickle

dump_dir = os.path.expanduser('dumps/')

ml_dataset = pickle.load(
    open(os.path.join(dump_dir, 'Dataset/ml-20m-first-10000'), 'rb'))
lt_dataset = pickle.load(
    open(os.path.join(dump_dir, 'Dataset/lt-first-10000'), 'rb'))

# ml_dataset.cut(limits=100)
# lt_dataset.cut(limits=100)

d1 = ml_dataset.info()
d2 = lt_dataset.info()
tag_set1 = d1.tags_set
tag_set2 = d2.tags_set
all_tag_set = tag_set1.intersection(tag_set2)
print('tags overlapping: {}'.format(len(all_tag_set)))

# param_grid = {'biased': [False], 'n_factors': [100],
#               'lr_all': [0.002, 0.005, 0.01], 'reg_all': [0.01, 0.02], 'n_epochs': [50]}

param_grid_with_topic_ml = {'biased': [False], 'n_factors': [100],
                            'lr_all': [0.002, 0.005, 0.01], 'reg_all': [0.01, 0.02, 0.04], 'n_epochs': [50],
                            'n_lda_iter': [2000], 'alpha': [0.01, 0.02, 0.04], 'eta': [0.01, 0.02]}

param_grid_with_topic_lt = {'biased': [False], 'n_factors': [100],
                         'lr_all': [0.02, 0.04], 'reg_all': [0.01, 0.02], 'n_epochs': [50], 
                         'n_lda_iter': [2000], 'alpha': [0.01, 0.02, 0.04, 0.1], 'eta': [0.01, 0.02]}

# grid_search1 = GridSearch(SVD, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info=dump_info)
# grid_search2 = GridSearch(UserItemTags, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info=dump_info)
# grid_search3 = GridSearch(ItemRelTags, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info=dump_info)
# grid_search4 = GridSearch(ItemTopics, param_grid_with_topic, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info=dump_info)
grid_search5 = GridSearch(CrossItemTopics, param_grid_with_topic_ml, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='ml_best_params_0520')
grid_search6 = GridSearch(CrossItemTopics, param_grid_with_topic_lt, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='lt_best_params_0520')


# grid_search1.evaluate(ml_dataset)
# grid_search2.evaluate(ml_dataset)
# grid_search3.evaluate(ml_dataset)
# grid_search4.evaluate(ml_dataset)
# grid_search5.evaluate(ml_dataset, aux_dataset=lt_dataset)
grid_search6.evaluate(lt_dataset, aux_dataset=ml_dataset)

# print("----CrossItemTopics-ml----")
# grid_search5.print_perf()
print("----CrossItemTopics-lt----")
grid_search6.print_perf()
