
from surprise import Dataset
from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags
from surprise import GridSearch
from surprise.chart import *

import os
import numpy as np
import pandas as pd
import pickle

# # dataset
# dataset_path1 = os.path.expanduser('./Dataset/ml-20m/')
# dataset_path2 = os.path.expanduser('./Dataset/LT/')

# # ml_dataset = Dataset(dataset_path=dataset_path1, tag_genome=False)
# lt_dataset = Dataset(dataset_path=dataset_path2,
#                      tag_genome=False, LT=True)
# lt_dataset.split(n_folds=5)
# lt_dataset.info()

dump_dir = os.path.expanduser('~') + '/Thesis/experiment/dumps'

ml_dataset = pickle.load(open(os.path.join(dump_dir, 'Dataset/ml-20m-first-10000'), 'rb'))

ml_dataset.info()


# param_grid = {'n_factors': [20, 40, 60, 80, 100,], 'lr_all': [
# 0.005, ], 'reg_all': [0.02, ], 'n_epochs': [50, ], 'n_lda_iter':[1000,],
# 'n_topics':[10,], 'alpha':[0.01, ], 'eta':[0.01, ]}

param_grid = {'n_factors': np.arange(10,210,10), 'lr_all': [
    0.005, ], 'reg_all': [0.02, ], 'n_epochs': [50, ]}

# grid_search0 = GridSearch(ItemRelTags, param_grid, measures=['RMSE', 'MAE'])
# grid_search1 = GridSearch(SVD, param_grid, measures=['RMSE', 'MAE'])
# grid_search2 = GridSearch(UserItemTags, param_grid, measures=['RMSE', 'MAE'])
grid_search3 = GridSearch(ItemRelTags, param_grid, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='factors_10_200')
grid_search4 = GridSearch(ItemTopics, param_grid, measures=[
                          'RMSE', 'MAE'], with_dump=True, dump_info='factors_10_200')

# grid_search0.evaluate(data)
# grid_search1.evaluate(data)
# grid_search2.evaluate(data)
grid_search3.evaluate(ml_dataset)
grid_search4.evaluate(ml_dataset)

# print("----ItemRelTags----")
# grid_search0.print_perf()
# print("----SVD----")
# grid_search1.print_perf()
# print('----UserItemTags----')
# grid_search2.print_perf()
print('----ItemRelTags----')
grid_search3.print_perf()
print("----ItemTopics----")
grid_search4.print_perf()


# results_df_3 = pd.DataFrame.from_dict(grid_search3.cv_results)


# results_df_4 = pd.DataFrame.from_dict(grid_search4.cv_results)
# print(results_df_4)
