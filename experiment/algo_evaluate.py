
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate, print_perf, evaluate_parts

import pickle
import os

from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags, CrossUserItemRelTags, CrossItemRelTags, CrossItemTopics

from surprise import GridSearch

from surprise.chart import *


# path to dataset file
dump_dir = os.path.expanduser('~') + '/Thesis/experiment/dumps/Dataset'

ml_dataset = pickle.load(open(os.path.join(dump_dir, 'ml-20m-first-10000'), 'rb'))
lt_dataset = pickle.load(open(os.path.join(dump_dir, 'lt-first-10000'), 'rb'))

d1 = ml_dataset.info()
d2 = lt_dataset.info()
tag_set1 = d1.tags_set
tag_set2 = d2.tags_set
all_tag_set = tag_set1.intersection(tag_set2)
print('tags overlapping: {}'.format(len(all_tag_set)))

# ml_dataset.cut(limits=100)
# lt_dataset.cut(limits=100)

algo1 = SVD(biased=False, n_factors=100, n_epochs=50, lr_all=0.005, reg_all=0.01)

algo2 = ItemTopics(biased=False, n_factors=100,
                   n_epochs=50, lr_all=0.005, reg_all=0.01, n_topics=10, n_lda_iter=2000, eta=0.01, alpha=0.02)

algo3 = ItemRelTags(biased=False, n_factors=100,
                    n_epochs=50, lr_all=0.005, reg_all=0.01)

algo4 = CrossItemTopics(biased=False, n_factors=100, n_epochs=50, lr_all=0.005,
                        reg_all=0.01, n_topics=10, n_lda_iter=2000, eta=0.01, alpha=0.02)

algo5 = UserItemTags(biased=False, n_factors=100, n_epochs=50, lr_all=0.005, reg_all=0.01)
algo6 = UserItemRelTags(biased=False, n_factors=100, n_epochs=50, lr_all=0.005, reg_all=0.01)
# Evaluate performances of our algorithm on the dataset.
print_perf(evaluate(algo1, dataset=ml_dataset, measures=['RMSE', 'MAE']))
print_perf(evaluate(algo2, dataset=ml_dataset, measures=['RMSE', 'MAE']))
print_perf(evaluate(algo3, dataset=ml_dataset, measures=['RMSE', 'MAE']))
print_perf(evaluate(algo4, dataset=ml_dataset, aux_dataset=lt_dataset, measures=['RMSE', 'MAE']))
print_perf(evaluate(algo5, dataset=ml_dataset, measures=['RMSE', 'MAE']))
print_perf(evaluate(algo6, dataset=ml_dataset, measures=['RMSE', 'MAE']))

# perf1 = evaluate_parts(algo1, dataset=ml_dataset, measures=[
#                        'RMSE', 'MAE'], trainset_parts=5, with_dump=True, dump_info='lt-unb-glo-5parts')
# perf2 = evaluate_parts(algo2, dataset=ml_dataset, measures=[
#                        'RMSE', 'MAE'], trainset_parts=5, with_dump=True, dump_info='lt-unb-glo-5parts')
# perf3 = evaluate_parts(algo3, dataset=ml_dataset, measures=[
#                        'RMSE', 'MAE'], trainset_parts=5, with_dump=True, dump_info='lt-unb-glo-5parts')
# perf4 = evaluate_parts(algo4, dataset=ml_dataset, aux_dataset=lt_dataset, measures=[
#                        'RMSE', 'MAE'], trainset_parts=5, with_dump=True, dump_info='lt-unb-glo-5parts')

# compare_part_usage_perf([perf1, perf2, perf3])
