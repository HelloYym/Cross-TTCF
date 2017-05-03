from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt

from six.moves import range
from six import iteritems

from .Reader import RatingsReader, TagsReader

from bidict import bidict
import time


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
        self.construct_trainset(raw_trainset)
        self.rating_scale = rating_scale
        self.offset = offset
        self._global_mean = None
        self._genome_tid = genome_tid
        self._genome_score = genome_score
        if genome_tid:
            # genome tag编号从1开始，预留一个0
            self.n_genome_tags = len(genome_tid) + 1
        self._item_tag_freq = self.cal_item_tag_freq()

    def construct_trainset(self, raw_trainset):

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
        for urid, irid, r, tags_list in raw_trainset:
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
        self.n_users = len(ur)  # number of users
        self.n_items = len(ir)  # number of items
        self.n_tags = current_t_index
        self.n_ratings = len(raw_trainset)

    def info(self, diagram=False):
        '''训练集统计信息'''

        print("Total number of ratings: {}".format(self.n_ratings))
        print("Unique users: {}".format(self.n_users))
        print("Unique items: {}".format(self.n_items))
        print("Unique tags: {}".format(self.n_tags))
        print("Tag assignments: {}".format(self.tag_assignments))
        print("Average ratings per user: {}".format(
            self.n_ratings / self.n_users))
        print("Average tags per rating: {}".format(
            self.tag_assignments / self.n_ratings))

    def cal_item_tag_freq(self):
        ''' cal the relevant tag occurrences for item.

        '''
        item_tag_freq = defaultdict(lambda: defaultdict(int))

        for _, iid, _, tids in self.uirts:
            for tid in tids:
                item_tag_freq[iid][tid] += 1

        # -1表示item的所有标签总个数
        for iid, tags_freq in item_tag_freq.items():
            item_tag_freq[iid][-1] = sum([freq for tid, freq in tags_freq.items()])

        return item_tag_freq

    def knows_user(self, uid):
        """Indicate if the user is part of the trainset.

        A user is part of the trainset if the user has at least one rating.

        Args:
            uid(int): The (inner) user id. See :ref:`this
                note<raw_inner_note>`.
        Returns:
            ``True`` if user is part of the trainset, else ``False``.
        """

        return uid in self.ur

    def knows_item(self, iid):
        """Indicate if the item is part of the trainset.

        An item is part of the trainset if the item was rated at least once.

        Args:
            iid(int): The (inner) item id. See :ref:`this
                note<raw_inner_note>`.
        Returns:
            ``True`` if item is part of the trainset, else ``False``.
        """

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
        return self._item_tag_freq[iiid][tid]

    def get_item_tags(self, iiid):
        return self._item_tag_freq[iiid]

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