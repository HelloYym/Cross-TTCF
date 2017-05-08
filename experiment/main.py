from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate, print_perf

import os
from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags
from surprise import GridSearch


# path to dataset file
dataset_path1 = os.path.expanduser('./Dataset/ml-latest-small/')
dataset_path2 = os.path.expanduser('./Dataset/LT/')
dataset1 = Dataset(dataset_path=dataset_path1, tag_genome=False)
dataset2 = Dataset(dataset_path=dataset_path2, tag_genome=False, LT=True, limits=502)
# dataset.split(n_folds=5)
d1 = dataset1.info()
d2 = dataset2.info()

tag_set1 = d1.tags_set
tag_set2 = d2.tags_set

all_tag_set = tag_set1.intersection(tag_set2)

print('tags overlapping: {}'.format(len(all_tag_set)))

# algo1 = SVD(biased=True, n_factors=15, n_epochs=20, lr_all=0.01)
# algo2 = UserItemTopics(biased=True, n_factors=15, n_epochs=20, lr_all=0.01, n_topics=5, n_lda_iter=500, verbose=True)
# algo3 = UserItemRelTags(biased=True, n_factors=15, n_epochs=20, lr_all=0.01)

algo4 = CrossUserItemTags(biased=True, n_factors=15, n_epochs=20, lr_all=0.01)
print_perf(evaluate(algo4, dataset=dataset1, aux_dataset=dataset2, measures=['RMSE', 'MAE']))

# # Evaluate performances of our algorithm on the dataset.
# print_perf(evaluate(algo1, dataset, measures=['RMSE', 'MAE']))
# print_perf(evaluate(algo2, dataset, measures=['RMSE', 'MAE']))
# print_perf(evaluate(algo3, dataset, measures=['RMSE', 'MAE']))
