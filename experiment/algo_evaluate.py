from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate, print_perf

import os
from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags
from surprise import GridSearch


# path to dataset file
dataset_path = os.path.expanduser('./Dataset/ml-20m/')
dataset = Dataset(dataset_path=dataset_path, tag_genome=False)
dataset.split(n_folds=5)
dataset.info()



algo = UserItemGenomeTags(biased=True, n_factors=100, n_epochs=40, lr_all=0.005, reg_all=0.02)

# Evaluate performances of our algorithm on the dataset.
print_perf(evaluate(algo, dataset, measures=['RMSE', 'MAE']))
