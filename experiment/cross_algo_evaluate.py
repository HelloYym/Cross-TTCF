from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate, print_perf

import os

from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags

from surprise import GridSearch


# # path to dataset file
# dataset_path = os.path.expanduser('./Dataset/ml-20m/')
# dataset = Dataset(dataset_path=dataset_path, tag_genome=False)
# dataset.split(n_folds=5)
# dataset.info()

# path to dataset file
dataset_path1 = os.path.expanduser('./Dataset/ml-20m/')
dataset_path2 = os.path.expanduser('./Dataset/LT/')
dataset1 = Dataset(dataset_path=dataset_path1, tag_genome=False)
dataset2 = Dataset(dataset_path=dataset_path2,
                   tag_genome=False, LT=True, limits=126083)
# dataset.split(n_folds=5)
d1 = dataset1.info()
d2 = dataset2.info()

tag_set1 = d1.tags_set
tag_set2 = d2.tags_set

all_tag_set = tag_set1.intersection(tag_set2)

print('tags overlapping: {}'.format(len(all_tag_set)))

algo = CrossUserItemTags(biased=True, n_factors=100,
                         n_epochs=50, lr_all=0.005, reg_all=0.02)

# Evaluate performances of our algorithm on the dataset.
print_perf(evaluate(algo, dataset=dataset2,
                    aux_dataset=dataset1, measures=['RMSE', 'MAE']))
