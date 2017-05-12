from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate, print_perf


import pickle
import os

from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags

from surprise import GridSearch


# path to dataset file
dataset_path1 = os.path.expanduser('./Dataset/ml-latest-small/')
# dataset_path2 = os.path.expanduser('./Dataset/LT/')

ml_dataset = Dataset(dataset_path=dataset_path1, tag_genome=False)
# lt_dataset = Dataset(dataset_path=dataset_path2,
#                      tag_genome=False, LT=True, limits=126083)


dump_dir = os.path.expanduser('~') + '/Thesis/experiment/dumps/Dataset'


pickle.dump(ml_dataset, open(os.path.join(dump_dir, 'ml-small'), 'wb'))
# pickle.dump(lt_dataset, open(os.path.join(dump_dir, 'LT-limit-ml'), 'wb'))


# ml_dataset = pickle.load(open(file_name, 'rb'))
# ml_dataset.split(n_folds=5)
# ml_dataset.info()

# algo = SVD(biased=True, n_factors=100,
#                        n_epochs=50, lr_all=0.005, reg_all=0.02)

# # Evaluate performances of our algorithm on the dataset.
# print_perf(evaluate(algo, dataset=ml_dataset, measures=['RMSE', 'MAE']))
