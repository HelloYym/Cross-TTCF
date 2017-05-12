from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate, print_perf, evaluate_parts


import pickle
import os

from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags

from surprise import GridSearch


# path to dataset file
dump_dir = os.path.expanduser('~') + '/Thesis/experiment/dumps/Dataset'
dataset_file = os.path.join(dump_dir, 'ml-small')

ml_dataset = pickle.load(open(dataset_file, 'rb'))


ml_dataset.split(n_folds=5)
ml_dataset.info()

algo = SVD(biased=True, n_factors=100,
           n_epochs=50, lr_all=0.005, reg_all=0.02)

# Evaluate performances of our algorithm on the dataset.
print_perf(evaluate(algo, dataset=ml_dataset, measures=['RMSE', 'MAE'], verbose=True))

# evaluate_parts(algo, dataset=ml_dataset, measures=['RMSE', 'MAE'], trainset_parts=10)
