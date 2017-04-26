import os
from surprise import Dataset
from surprise import SVD, UserItemTags, UserItemGenomeTags
from surprise import GridSearch


# dataset
dataset_path = os.path.expanduser('./Dataset/ml-20m/')
data = Dataset(dataset_path=dataset_path, tag_genome=True)
data.split(n_folds=5)
data.info()


param_grid = {'n_factors': [120,], 'lr_all': [
    0.005, ], 'reg_all': [0.02,], 'n_epochs': [40, ]}

# grid_search1 = GridSearch(SVD, param_grid, measures=['RMSE', 'MAE'])
grid_search2 = GridSearch(UserItemTags, param_grid, measures=['RMSE', 'MAE'])
grid_search3 = GridSearch(UserItemGenomeTags, param_grid, measures=['RMSE', 'MAE'])

# grid_search1.evaluate(data)
grid_search2.evaluate(data)
grid_search3.evaluate(data)

# print("----SVD----")
# grid_search1.print_perf()
print('----UserItemTags----')
grid_search2.print_perf()
print('----UserItemGenomeTags----')
grid_search3.print_perf()

