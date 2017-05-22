from surprise import KNNWithMeans
from surprise import Dataset
from surprise import evaluate, print_perf
import pickle
import os
from surprise import SVD, UserItemTags, UserItemGenomeTags, ItemRelTags, UserItemRelTags, ItemTopics, UserItemTopics
from surprise import CrossUserItemTags, CrossUserItemRelTags
from surprise import GridSearch
from surprise.chart import *


# path to dataset file
dump_dir = os.path.expanduser('dumps/')
ml_dataset = pickle.load(
    open(os.path.join(dump_dir, 'Dataset/ml-20m'), 'rb'))
lt_dataset = pickle.load(
    open(os.path.join(dump_dir, 'Dataset/lt-limit-ml20m'), 'rb'))


uirts = lt_dataset.info().uirts
tag_freq = defaultdict(int)
for _, _, _, tids in uirts:
    for tid in tids:
        tag_freq[tid] += 1

tag_power_law(tag_freq)

# lt_dataset.info()

# tag_set1 = d1.tags_set
# tag_set2 = d2.tags_set

# all_tag_set = tag_set1.intersection(tag_set2)

# print('tags overlapping: {}'.format(len(all_tag_set)))
