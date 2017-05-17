import os
import pickle

from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags, CrossUserItemRelTags, CrossItemRelTags, CrossItemTopics

from surprise import GridSearch
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate, print_perf, evaluate_parts
from surprise.chart import *


dump_dir = os.path.expanduser('~') + '/Thesis/experiment/dumps/Dataset'

ml_dataset = pickle.load(open(os.path.join(dump_dir, 'ml-20m-first-10000'), 'rb'))
lt_dataset = pickle.load(open(os.path.join(dump_dir, 'lt-first-10000'), 'rb'))

d1 = ml_dataset.info()
d2 = lt_dataset.info()
tag_set1 = d1.tags_set
tag_set2 = d2.tags_set
all_tag_set = tag_set1.intersection(tag_set2)
print('tags overlapping: {}'.format(len(all_tag_set)))


# algo_single = ItemRelTags(biased=True, n_factors=50,
#                           n_epochs=100, lr_all=0.002, reg_all=0.01)

algo_cross = CrossItemTopics(biased=True, n_factors=100, n_epochs=100, lr_all=0.001,
                             reg_all=0.01, n_topics=10, n_lda_iter=1000, eta=0.01, alpha=0.02)


# print_perf(evaluate(algo_cross, dataset=ml_dataset,
#                     aux_dataset=lt_dataset, measures=['RMSE', 'MAE']))


evaluate_parts(algo_cross, dataset=ml_dataset, aux_dataset=lt_dataset, measures=[
    'RMSE', 'MAE'], trainset_parts=5, with_dump=True, dump_info='ml-100f-0001lr-5parts')
