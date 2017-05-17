
from surprise import Dataset
from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags, CrossUserItemRelTags, CrossItemRelTags, CrossItemTopics
from surprise import GridSearch
from surprise.chart import *

import os
import numpy as np
import pandas as pd
import pickle

dump_dir = os.path.expanduser('~') + '/Thesis/experiment/dumps'

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


# param_grid = {'n_factors': [20, 40, 60, 80, 100,], 'lr_all': [
# 0.005, ], 'reg_all': [0.02, ], 'n_epochs': [50, ], 'n_lda_iter':[1000,],
# 'n_topics':[10,], 'alpha':[0.01, ], 'eta':[0.01, ]}

# param_grid = {'n_factors': [100], 'lr_all': [0.001, 0.002], 'reg_all': [0.01, ], 'n_epochs': [
#     50, ], 'n_topics': [10, 20], 'alpha': [0.01, 0.02], 'eta': [0.01, 0.02]}

param_grid = {'biased': [False, ], 'n_factors': [100], 'lr_all': [
    0.001, 0.002, 0.005, ], 'reg_all': [0.01, 0.02, 0.04], 'n_epochs': [50]}

grid_search0 = GridSearch(SVD, param_grid, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-ml')
grid_search1 = GridSearch(UserItemTags, param_grid, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-ml')
grid_search2 = GridSearch(UserItemRelTags, param_grid, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-ml')
grid_search3 = GridSearch(ItemRelTags, param_grid, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-ml')

grid_search01 = GridSearch(SVD, param_grid, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-lt')
grid_search11 = GridSearch(UserItemTags, param_grid, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-lt')
grid_search21 = GridSearch(UserItemRelTags, param_grid, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-lt')
grid_search31 = GridSearch(ItemRelTags, param_grid, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-lt')

# grid_search4 = GridSearch(ItemTopics, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-ml')
# grid_search5 = GridSearch(ItemTopics, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-lt')

# grid_search6 = GridSearch(CrossItemTopics, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-ml')
# grid_search7 = GridSearch(CrossItemTopics, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-lt')

grid_search0.evaluate(ml_dataset)
grid_search1.evaluate(ml_dataset)
grid_search2.evaluate(ml_dataset)
grid_search3.evaluate(ml_dataset)

grid_search01.evaluate(lt_dataset)
grid_search11.evaluate(lt_dataset)
grid_search21.evaluate(lt_dataset)
grid_search31.evaluate(lt_dataset)
# grid_search4.evaluate(ml_dataset)
# grid_search5.evaluate(lt_dataset)
# grid_search6.evaluate(ml_dataset, aux_dataset=lt_dataset)
# grid_search7.evaluate(lt_dataset, aux_dataset=ml_dataset)

print("----SVD----")
grid_search0.print_perf()
print("----UserItemTags----")
grid_search1.print_perf()
print('----UserItemRelTags----')
grid_search2.print_perf()
print('----ItemRelTags----')
grid_search3.print_perf()

print("----SVD-lt----")
grid_search01.print_perf()
print("----UserItemTags-lt----")
grid_search11.print_perf()
print('----UserItemRelTags-lt----')
grid_search21.print_perf()
print('----ItemRelTags-lt----')
grid_search31.print_perf()

# print("----ItemTopics-ml----")
# grid_search4.print_perf()
# print("----ItemTopics-lt----")
# grid_search5.print_perf()
# print("----CrossItemTopics-ml----")
# grid_search6.print_perf()
# print("----CrossItemTopics-lt----")
# grid_search7.print_perf()


# results_df_3 = pd.DataFrame.from_dict(grid_search3.cv_results)


# results_df_4 = pd.DataFrame.from_dict(grid_search4.cv_results)
# print(results_df_4)
