import os
from surprise import Dataset
from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags, CrossUserItemRelTags
from surprise import GridSearch


# dataset
dataset_path1 = os.path.expanduser('./Dataset/ml-20m/')
dataset_path2 = os.path.expanduser('./Dataset/LT/')

# ml_dataset = Dataset(dataset_path=dataset_path1, tag_genome=False)
lt_dataset = Dataset(dataset_path=dataset_path2,
                     tag_genome=False, LT=True)

lt_dataset.split(n_folds=5)
lt_dataset.info()


param_grid = {'n_factors': [100, ], 'lr_all': [
    0.005, ], 'reg_all': [0.02, ], 'n_epochs': [50, ]}

grid_search0 = GridSearch(ItemRelTags, param_grid, measures=['RMSE', 'MAE'])
grid_search1 = GridSearch(SVD, param_grid, measures=['RMSE', 'MAE'])
grid_search2 = GridSearch(UserItemTags, param_grid, measures=['RMSE', 'MAE'])
grid_search3 = GridSearch(UserItemRelTags, param_grid, measures=['RMSE', 'MAE'])
grid_search4 = GridSearch(ItemTopics, param_grid, measures=['RMSE', 'MAE'])


grid_search0.evaluate(lt_dataset)
grid_search1.evaluate(lt_dataset)
grid_search2.evaluate(lt_dataset)
grid_search3.evaluate(lt_dataset)
grid_search4.evaluate(lt_dataset)

print("----ItemRelTags----")
grid_search0.print_perf()
print("----SVD----")
grid_search1.print_perf()
print('----UserItemTags----')
grid_search2.print_perf()
print('----UserItemRelTags----')
grid_search3.print_perf()
print('----ItemTopics----')
grid_search4.print_perf()