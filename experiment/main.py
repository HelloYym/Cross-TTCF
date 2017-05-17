from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate, print_perf
import pickle
import os
from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags, CrossUserItemRelTags
from surprise import GridSearch


# path to dataset file
dump_dir = os.path.expanduser('~') + '/Thesis/experiment/dumps/Dataset'
dataset_file = os.path.join(dump_dir, 'ml-20m')
ml_dataset = pickle.load(open(dataset_file, 'rb'))
ml_dataset.cut(10000)
ml_dataset.info()
# d2 = dataset2.info()

# tag_set1 = d1.tags_set
# tag_set2 = d2.tags_set

# all_tag_set = tag_set1.intersection(tag_set2)

# print('tags overlapping: {}'.format(len(all_tag_set)))

# algo1 = SVD(biased=True, n_factors=15, n_epochs=20, lr_all=0.01)
algo2 = ItemTopics(biased=True, n_factors=100, n_epochs=50, lr_all=0.005, reg_all=0.02)
algo3 = ItemRelTags(biased=True, n_factors=20, n_epochs=20, lr_all=0.01)

# algo4 = CrossUserItemTags(biased=True, n_factors=15, n_epochs=20, lr_all=0.01)
# print_perf(evaluate(algo4, dataset=dataset1, aux_dataset=dataset2, measures=['RMSE', 'MAE']))

# # Evaluate performances of our algorithm on the dataset.
# print_perf(evaluate(algo1, dataset, measures=['RMSE', 'MAE']))
print_perf(evaluate(algo2, ml_dataset, measures=['RMSE', 'MAE']))
print_perf(evaluate(algo3, ml_dataset, measures=['RMSE', 'MAE']))
