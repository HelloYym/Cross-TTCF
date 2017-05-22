
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt

from six.moves import range
from six import iteritems

from .Reader import RatingsReader, TagsReader

from bidict import bidict
import time
import lda
from scipy import sparse


class Trainset:
    """A trainset contains all useful data that constitutes a training set.


    Attributes:
        ur(:obj:`defaultdict` of :obj:`list`): The users ratings. This is a
            dictionary containing lists of tuples of the form ``(item_inner_id,
            rating)``. The keys are user inner ids.
        ir(:obj:`defaultdict` of :obj:`list`): The items ratings. This is a
            dictionary containing lists of tuples of the form ``(user_inner_id,
            rating)``. The keys are item inner ids.
        n_users: Total number of users :math:`|U|`.
        n_items: Total number of items :math:`|I|`.
        n_tags: Total number of tags
        n_ratings: Total number of ratings :math:`|R_{train}|`.
        rating_scale(tuple): The minimum and maximal rating of the rating
            scale.
        global_mean: The mean of all ratings :math:`\\mu`.
    """

    def __init__(self, raw_trainset, rating_scale, offset, genome_tid, genome_score):

        self.raw_trainset = raw_trainset

        self.tags_set = self.get_tags_set(self.raw_trainset)
        self.construct()
        self.rating_scale = rating_scale
        self.offset = offset
        self._global_mean = None
        self._genome_tid = genome_tid
        self._genome_score = genome_score
        if genome_tid:
            # genome tag编号从1开始，预留一个0
            self.n_genome_tags = len(genome_tid) + 1

    def get_tags_set(self, uirts):
        tags_set = set()
        for _, _, _, tags in uirts:
            for tag in tags:
                tags_set.add(tag)
        return tags_set

    def construct(self):

        raw2inner_id_users = bidict()
        raw2inner_id_items = bidict()
        raw2inner_id_tags = bidict()

        current_u_index = 0
        current_i_index = 0
        current_t_index = 0

        tag_assignments = 0

        ur = defaultdict(list)
        ir = defaultdict(list)
        uirts = list()

        # user raw id, item raw id, translated rating, tags
        for urid, irid, r, tags_list in self.raw_trainset:
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            # 在trainset中保存inner_tid
            tids = list()
            for tag in tags_list:
                if tag in self.tags_set:
                    try:
                        tid = raw2inner_id_tags[tag]
                    except KeyError:
                        tid = current_t_index
                        raw2inner_id_tags[tag] = current_t_index
                        current_t_index += 1
                    tids.append(tid)

            tag_assignments += len(tids)    # 统计所有的tag数目
            ur[uid].append((iid, r, tids))
            ir[iid].append((uid, r, tids))
            uirts.append((uid, iid, r, tids))

        self._raw2inner_id_users = raw2inner_id_users
        self._raw2inner_id_items = raw2inner_id_items
        self._raw2inner_id_tags = raw2inner_id_tags
        self.tag_assignments = tag_assignments
        self.uirts = uirts
        self.ur = ur
        self.ir = ir
        self.n_users = current_u_index  # number of users
        self.n_items = current_i_index  # number of items
        self.n_tags = current_t_index
        self.n_ratings = len(self.raw_trainset)
        self.item_tag_freq = self.cal_item_tag_freq()

    def rank_sum_test(self, confidence=0.95):
        # 具有某个tag的rating数肯定不会超过50%

        # 先计算每一个分数的秩
        # 每个rating的个数
        ratings_num = defaultdict(int)
        for _, _, r, tags in self.uirts:
            ratings_num[r] += 1
        # 按中位数作为秩
        ratings_median = defaultdict(int)
        c = 0
        for r in np.arange(0.5, 5.5, 0.5):
            ratings_median[r] = c + (ratings_num[r] + 1) * 0.5
            c += ratings_num[r]

        # 每个tag的秩list
        tag_ranks_dict = defaultdict(list)
        for _, _, r, tags in self.uirts:
            ratings_num[r] += 1
            for tag in tags:
                tag_ranks_dict[tag].append(ratings_median[r])

        all_size = len(self.uirts)
        for tag, tag_ranks in tag_ranks_dict.items():
            m = len(tag_ranks)
            n = all_size - m
            # tag的秩和
            rank_sum = sum(tag_ranks)
            # 进行rank-sum test
            halfMsum = 0.5 * m * (m + n + 1)
            twelthMNsum = (1.0 / 6) * halfMsum * n
            zNumerator = rank_sum - halfMsum
            zDenominator = twelthMNsum ** 0.5
            z = abs(zNumerator / zDenominator)
            # 如果结果小于置信度，说明该tag对评分影响不大，删除该评分
            if z < confidence:
                self.tags_set.discard(self.to_raw_tag(tag))

    def info(self, diagram=False):
        '''训练集统计信息'''
        print('#' * 12)
        print("Total number of ratings: {}".format(self.n_ratings))
        print("Unique users: {}".format(self.n_users))
        print("Unique items: {}".format(self.n_items))
        print("Unique tags: {}".format(self.n_tags))
        print("Tag assignments: {}".format(self.tag_assignments))
        print("Average ratings per user: {}".format(
            self.n_ratings / self.n_users))
        print("Average tags per rating: {}".format(
            self.tag_assignments / self.n_ratings))
        print("Data sparse: {}".format(format(1 - self.n_ratings /
                                       (self.n_users * self.n_items), '.2%')))
        print('#' * 12)

    def cal_item_tag_freq(self):
        ''' cal the relevant tag occurrences for item.

        '''
        item_tag_freq = defaultdict(lambda: defaultdict(int))
        for _, iid, _, tids in self.uirts:
            for tid in tids:
                item_tag_freq[iid][tid] += 1
        return item_tag_freq

    def knows_user(self, uid):
        return uid in self.ur

    def knows_item(self, iid):

        return iid in self.ir

    def knows_tag(self, tag):
        return tag in self._raw2inner_id_tags

    def to_inner_uid(self, ruid):
        try:
            return self._raw2inner_id_users[ruid]
        except KeyError:
            raise ValueError(('raw user id ' + str(ruid) +
                              ' is not part of the trainset.'))

    def to_raw_uid(self, iuid):
        try:
            return self._raw2inner_id_users.inv[iuid]
        except KeyError:
            raise ValueError(('inner item id ' + str(iuid) +
                              ' is not part of the trainset.'))

    def to_inner_iid(self, riid):
        try:
            return self._raw2inner_id_items[riid]
        except KeyError:
            raise ValueError(('raw item id ' + str(riid) +
                              ' is not part of the trainset.'))

    def to_raw_iid(self, iiid):
        try:
            return self._raw2inner_id_items.inv[iiid]
        except KeyError:
            raise ValueError(('inner item id ' + str(iiid) +
                              ' is not part of the trainset.'))

    def to_inner_tid(self, rtag):
        try:
            return self._raw2inner_id_tags[rtag]
        except KeyError:
            raise ValueError(('raw tag id ' + str(rtag) +
                              ' is not part of the trainset.'))

    def to_raw_tag(self, itid):
        try:
            return self._raw2inner_id_tags.inv[itid]
        except KeyError:
            raise ValueError(('inner tag id ' + str(itid) +
                              ' is not part of the trainset.'))

    def to_genome_tid(self, tag):
        try:
            return self._genome_tid[tag]
        except KeyError:
            raise ValueError(('Tag ' + str(tag) + ' is not the genome tag.'))

    def is_genome_tag(self, tag):
        '''判断一个tag是否在genome列表中

        '''
        return tag in self._genome_tid

    def get_genome_score(self, riid, gtid):
        try:
            genome_score = self._genome_score[(riid, gtid)]
        except:
            genome_score = 0
        return genome_score

    def get_item_tag_freq(self, iiid, tid):
        return self.item_tag_freq[iiid][tid]

    def get_item_tags(self, iiid):
        return self.item_tag_freq[iiid]

    def all_ratings(self):
        """Generator function to iterate over all ratings.

        Yields:
            A tuple ``(uid, iid, rating)`` where ids are inner ids.
        """

        for u, i, r, _ in self.uirts:
            yield u, i, r

    def all_ratings_tags(self):
        """Generator function to iterate over all ratings.

        Yields:
            A tuple ``(uid, iid, rating)`` where ids are inner ids.
        """

        for u, i, r, tids in self.uirts:
            yield u, i, r, tids

    def all_ratings_genome_tags_score(self):
        """Generator function to iterate over all ratings.

        Yields:
            A tuple ``(uid, iid, rating)`` where ids are inner ids.
        """
        for u, i, r, tids in self.uirts:
                # 返回tags和genome关联度
            tags_score = [(self.to_genome_tid(self.to_raw_tag(tid)), self.get_genome_score(self.to_raw_iid(i), self.to_genome_tid(self.to_raw_tag(tid))))
                          for tid in tids if self.is_genome_tag(self.to_raw_tag(tid))]
            yield u, i, r, tags_score

    def all_users(self):
        """Generator function to iterate over all users.

        Yields:
            Inner id of users.
        """
        return range(self.n_users)

    def all_items(self):
        """Generator function to iterate over all items.

        Yields:
            Inner id of items.
        """
        return range(self.n_items)

    @property
    def global_mean(self):
        """Return the mean of all ratings.

        It's only computed once."""
        if self._global_mean is None:
            self._global_mean = np.mean([r for (_, _, r) in
                                         self.all_ratings()])

        return self._global_mean

    # def discard_less_common_tag(self, threshold):
    #     for u, i, r, tags in self.user_item_rating_tags:
    #         for tag in tags:
    #             if self.tag_freq[tag] < threshold:
    #                 self.tags_set.discard(tag)

    # def tag_cleaning(self):
    #     user_item_rating_tags = list()
    #     for u, i, r, tags in self.user_item_rating_tags:
    #         tags_cleanned = [tag for tag in tags if tag in self.tags_set]
    #         user_item_rating_tags.append((u, i, r, tags_cleanned))

    #     return user_item_rating_tags
