
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


param_grid1 = {'biased': [False, ], 'n_factors': [100],
               'lr_all': [0.005, ], 'reg_all': [0.01], 'n_epochs': [50]}
param_grid2 = {'biased': [False, ], 'n_factors': [100],
               'lr_all': [0.005, ], 'reg_all': [0.01], 'n_epochs': [50], 'confidence':[0, 0.4, 0.75, 0.95, 1.1]}


grid_search1 = GridSearch(UserItemTags, param_grid1, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='search_confidence-ml')
grid_search2 = GridSearch(UserItemRelTags, param_grid2, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='search_confidence-ml')
grid_search3 = GridSearch(ItemRelTags, param_grid2, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='search_confidence-ml')

# grid_search01 = GridSearch(SVD, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-lt')
# grid_search11 = GridSearch(UserItemTags, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-lt')
# grid_search21 = GridSearch(UserItemRelTags, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-lt')
# grid_search31 = GridSearch(ItemRelTags, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-lt')

# grid_search4 = GridSearch(ItemTopics, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-ml')
# grid_search5 = GridSearch(ItemTopics, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-lt')

# grid_search6 = GridSearch(CrossItemTopics, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-ml')
# grid_search7 = GridSearch(CrossItemTopics, param_grid, measures=[
#                           'RMSE', 'MAE'], with_dump=True, dump_info='search_best_perf-lt')


grid_search1.evaluate(ml_dataset)
grid_search2.evaluate(ml_dataset)
grid_search3.evaluate(ml_dataset)

# grid_search01.evaluate(lt_dataset)
# grid_search11.evaluate(lt_dataset)
# grid_search21.evaluate(lt_dataset)
# grid_search31.evaluate(lt_dataset)
# grid_search4.evaluate(ml_dataset)
# grid_search5.evaluate(lt_dataset)
# grid_search6.evaluate(ml_dataset, aux_dataset=lt_dataset)
# grid_search7.evaluate(lt_dataset, aux_dataset=ml_dataset)


print("----UserItemTags----")
grid_search1.print_perf()
print('----UserItemRelTags----')
grid_search2.print_perf()
print('----ItemRelTags----')
grid_search3.print_perf()

# print("----SVD-lt----")
# grid_search01.print_perf()
# print("----UserItemTags-lt----")
# grid_search11.print_perf()
# print('----UserItemRelTags-lt----')
# grid_search21.print_perf()
# print('----ItemRelTags-lt----')
# grid_search31.print_perf()

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
