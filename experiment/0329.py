from surprise import SVD, SVDpp, NMF, KNNBasic, KNNWithMeans, CoClustering, SlopeOne
from surprise import Dataset
from surprise import evaluate, print_perf
from kkn_test import KNNBasicTest
from surprise import Reader, RatingsReader, TagsReader
import os
import operator


# path to dataset file
ratings_file_path = os.path.expanduser('ml-latest-small/ratings.csv')
tags_file_path = os.path.expanduser('ml-latest-small/tags.csv')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
ratings_reader = RatingsReader(line_format='user item rating timestamp',
                               sep=',', rating_scale=(0.5, 5), skip_lines=1)
tags_reader = TagsReader(
    line_format='user item tags timestamp', sep=',', skip_lines=1)

data = Dataset(ratings_file_path, tags_file_path, ratings_reader, tags_reader)

# tags_set = {}

# for line in data.raw_tags:
# 	tags = line[2]
# 	for tag in tags:
# 		if tag not in tags_set:
# 			tags_set[tag] = 1
# 		else:
# 			tags_set[tag] += 1

# sorted_x = sorted(tags_set.items(), key=operator.itemgetter(1))

# for key, value in sorted_x:
# 	print(key, end=' ')
# 	print(value)


data.split(n_folds=3)

# # We'll use the famous SVD algorithm.
# algo1 = SVD(biased=True, n_factors=10, n_epochs=20, lr_all=.001, reg_all=.02)
algo2 = KNNBasicTest(k=200, sim_options={'user_based': False, 'name':'msd'})
# # algo3 = KNNWithMeans(sim_options={'user_based':False})

# # Evaluate performances of our algorithm on the dataset.
# print_perf(evaluate(algo1, data, measures=['RMSE', 'MAE']))
print_perf(evaluate(algo2, data, measures=['RMSE', 'MAE']))
# #evaluate(algo3, data, measures=['RMSE', 'MAE'])
