from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate, print_perf

import os

from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags

from surprise import GridSearch


# path to dataset file
dataset_path1 = os.path.expanduser('./Dataset/ml-20m/')
dataset_path2 = os.path.expanduser('./Dataset/LT/')

# ml_dataset = Dataset(dataset_path=dataset_path1, tag_genome=False)
lt_dataset = Dataset(dataset_path=dataset_path2,
                     tag_genome=False, LT=True)
lt_dataset.split(n_folds=5)
lt_dataset.info()


algo = UserItemRelTags(biased=True, n_factors=100,
                       n_epochs=50, lr_all=0.005, reg_all=0.02)

# Evaluate performances of our algorithm on the dataset.
print_perf(evaluate(algo, dataset=lt_dataset, measures=['RMSE', 'MAE']))
