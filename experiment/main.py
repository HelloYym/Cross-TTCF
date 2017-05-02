from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate, print_perf

import os
from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags
from surprise import GridSearch


# path to dataset file
dataset_path = os.path.expanduser('./Dataset/ml-latest-small/')
dataset = Dataset(dataset_path=dataset_path, tag_genome=False)
dataset.split(n_folds=5)
dataset.info()


# algo1 = SVD(biased=True, n_factors=15, n_epochs=20, lr_all=0.01)
# algo2 = ItemRelTags(biased=True, n_factors=15, n_epochs=20, lr_all=0.01)
# algo3 = UserItemTags(biased=True, n_factors=15, n_epochs=20, lr_all=0.01)


# # Evaluate performances of our algorithm on the dataset.
# # print_perf(evaluate(algo1, dataset, measures=['RMSE', 'MAE']))
# # print_perf(evaluate(algo2, dataset, measures=['RMSE', 'MAE']))
# print_perf(evaluate(algo3, dataset, measures=['RMSE', 'MAE']))
