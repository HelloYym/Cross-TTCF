from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt

from six.moves import range

from .Reader import RatingsReader, TagsReader, LTReader
from .trainset import Trainset
import lda


class Dataset:
    """Base class for loading datasets.

    Note that you should never instantiate the :class:`Dataset` class directly
    (same goes for its derived classes), but instead use one of the three
    available methods for loading datasets."""

    def __init__(self, dataset_path, sep=',', rating_scale=(0.5, 5), skip_lines=0,
                 tag_genome=False, limits=None, LT=False, first_n_ratings=None):

        if LT:
            self.reader = LTReader(txt_file=dataset_path +
                                   'UI2.txt', skip_lines=skip_lines)

            self.raw_ratings, self.raw_tags = self.reader.read()
        else:
            self.ratings_reader = RatingsReader(
                dataset_path=dataset_path, sep=sep,
                rating_scale=rating_scale, skip_lines=skip_lines)
            self.tags_reader = TagsReader(
                dataset_path=dataset_path, sep=sep, skip_lines=skip_lines)

            self.raw_tags = self.tags_reader.read()
            self.raw_ratings = self.ratings_reader.read()

        self.tag_genome = tag_genome
        self.genome_tid, self.genome_score = self.tags_reader.read_genome(
        ) if tag_genome else (None, None)

        self.n_folds = 5
        self.shuffle = True
        self.rating_scale = rating_scale

        # 只选取有标签的评分
        self.user_item_rating_tags = self.combine_rating_tag(first_n_ratings)
        del self.raw_ratings
        del self.raw_tags

        if self.shuffle:
            random.shuffle(self.user_item_rating_tags)
            self.shuffle = False  # set to false for future calls to raw_folds
        if limits:
            self.user_item_rating_tags = self.user_item_rating_tags[:limits]

    def cut(self, limits):
        self.user_item_rating_tags = self.user_item_rating_tags[
            :min(limits, len(self.user_item_rating_tags))]

    def combine_rating_tag(self, first_n_ratings):
        ''' we consider only the ratings in which at least one tag was used.self

        '''
        user_item_rating_tags = list()
        for user, item, rating in self.raw_ratings:
            if (user, item) in self.raw_tags:
                user_item_rating_tags.append(
                    [user, item, rating, self.raw_tags[(user, item)]])
                if first_n_ratings and len(user_item_rating_tags) >= first_n_ratings:
                    break
        print("there are {} ratings in which at least one tag was used.".format(
            len(user_item_rating_tags)))
        return user_item_rating_tags

    def raw_folds(self):

        # random.shuffle(self.user_item_rating_tags)

        def k_folds(seq, n_folds):
            """Inspired from scikit learn KFold method."""

            if n_folds > len(seq) or n_folds < 2:
                raise ValueError('Incorrect value for n_folds.')

            start, stop = 0, 0
            for fold_i in range(n_folds):
                start = stop
                stop += len(seq) // n_folds
                if fold_i < len(seq) % n_folds:
                    stop += 1
                yield seq[:start] + seq[stop:], seq[start:stop]

        return k_folds(self.user_item_rating_tags, self.n_folds)

    def folds(self, n_parts=10, total_parts=10):
        """Generator function to iterate over the folds of the Dataset.

        See :ref:`User Guide <iterate_over_folds>` for usage.

        Yields:
            tuple: :class:`Trainset` and testset of current fold.
        """

        for raw_trainset, raw_testset in self.raw_folds():
            partial_raw_trainset = raw_trainset[
                :int(len(raw_trainset) * n_parts / total_parts)]
            trainset = self.construct_trainset(partial_raw_trainset)
            testset = self.construct_testset(raw_testset)
            yield trainset, testset

    def construct_trainset(self, raw_trainset):

        trainset = Trainset(raw_trainset,
                            self.rating_scale,
                            0,
                            self.genome_tid,
                            self.genome_score)

        return trainset

    def construct_testset(self, raw_testset):

        return [(ruid, riid, r_ui_rating, r_ui_tags)
                for (ruid, riid, r_ui_rating, r_ui_tags) in raw_testset]

    def build_full_trainset(self):
        """Do not split the dataset into folds and just return a trainset as
        is, built from the whole dataset.

        """

        return self.construct_trainset(self.user_item_rating_tags)

    def split(self, n_folds=5, shuffle=True):
        """Split the dataset into folds for futur cross-validation.

        """

        self.n_folds = n_folds
        self.shuffle = shuffle

    def info(self, diagram=False):
        '''计算每个tag的出现频数'''

        dataset = self.build_full_trainset()
        # dataset.rank_sum_test(0.95)
        # dataset.construct()
        dataset.info()
        # X = np.zeros((dataset.n_items, dataset.n_tags), dtype=int)
        # for _, iid, _, tids in dataset.uirts:
        #     for tid in tids:
        #         X[iid, tid] += 1

        # vocab = [dataset.to_raw_tag(tid) for tid in range(dataset.n_tags)]
        # lda_model = lda.LDA(n_topics=10, n_iter=4000,
        #                     alpha=0.1, eta=0.01, refresh=1000)
        # lda_model.fit(X)

        # topic_word = lda_model.topic_word_
        # n_top_words = 15
        # for i, topic_dist in enumerate(topic_word):
        #     topic_words = np.array(vocab)[np.argsort(topic_dist)][
        #         :-n_top_words:-1]
        #     print('Topic {}: {}'.format(i, '; '.join(topic_words)))

        return dataset
