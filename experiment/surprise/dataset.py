
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt

from six.moves import range
from six import iteritems

from .Reader import RatingsReader, TagsReader

from bidict import bidict
import time


class Dataset:
    """Base class for loading datasets.

    Note that you should never instantiate the :class:`Dataset` class directly
    (same goes for its derived classes), but instead use one of the three
    available methods for loading datasets."""

    def __init__(self, dataset_path, sep=',', rating_scale=(0.5, 5), skip_lines=0,
                 tag_genome=False):

        self.ratings_reader = RatingsReader(
            dataset_path=dataset_path, sep=sep,
            rating_scale=rating_scale, skip_lines=skip_lines)

        self.tags_reader = TagsReader(
            dataset_path=dataset_path, sep=sep, skip_lines=skip_lines)

        self.n_folds = 5
        self.shuffle = True
        self.tag_genome = tag_genome
        self.raw_tags = self.tags_reader.read()
        self.genome_tid, self.genome_score = self.tags_reader.read_genome(
        ) if tag_genome else (None, None)
        self.raw_ratings = self.ratings_reader.read()

        # 只选取有标签的评分
        self.user_item_rating_tags = self.combine_rating_tag()

        # 进行数据清洗，例如去除低频标签、ranksum等
        self.tags_set = self.get_tags_set()
        self.tag_freq = self.cal_tag_freq()

        # self.discard_less_common_tag(threshold=10)
        self.rank_sum_test(confidence=0.95)
        self.user_item_rating_tags = self.tag_cleaning()

    def combine_rating_tag(self):
        ''' we consider only the ratings in which at least one tag was used.self

        '''
        user_item_rating_tags = list()
        for user, item, rating in self.raw_ratings:
            if (user, item) in self.raw_tags:
                user_item_rating_tags.append(
                    [user, item, rating, self.raw_tags[(user, item)]])
        print("there are {} ratings in which at least one tag was used.".format(
            len(user_item_rating_tags)))
        return user_item_rating_tags

    def get_tags_set(self):
        tags_set = set()
        for _, _, _, tags in self.user_item_rating_tags:
            for tag in tags:
                tags_set.add(tag)
        return tags_set

    def rank_sum_test(self, confidence=0.95):
        # 具有某个tag的rating数肯定不会超过50%

        # 先计算每一个分数的秩
        ratings_num = defaultdict(int)
        for _, _, r, tags in self.user_item_rating_tags:
            ratings_num[r] += 1

        ratings_median = defaultdict(int)
        c = 0
        for r in np.arange(0.5, 5.5, 0.5):
            ratings_median[r] = c + (ratings_num[r] + 1) * 0.5
            c += ratings_num[r]

        tag_rank_sum = defaultdict(list)
        for _, _, r, tags in self.user_item_rating_tags:
            ratings_num[r] += 1
            for tag in tags:
                tag_rank_sum[tag].append(ratings_median[r])

        all_size = len(self.user_item_rating_tags)
        for tag, tag_ranks in tag_rank_sum.items():
            m = len(tag_ranks)
            n = all_size - m
            rank_sum = sum(tag_ranks)

            halfMsum = 0.5 * m * (m + n + 1)
            twelthMNsum = (1.0 / 6) * halfMsum * n
            zNumerator = rank_sum - halfMsum
            zDenominator = twelthMNsum ** 0.5
            z = abs(zNumerator / zDenominator)

            if z < confidence:
                self.tags_set.discard(tag)

    def discard_less_common_tag(self, threshold):
        for u, i, r, tags in self.user_item_rating_tags:
            for tag in tags:
                if self.tag_freq[tag] < threshold:
                    self.tags_set.discard(tag)

    def tag_cleaning(self):
        user_item_rating_tags = list()
        for u, i, r, tags in self.user_item_rating_tags:
            tags_cleanned = [tag for tag in tags if tag in self.tags_set]
            user_item_rating_tags.append((u, i, r, tags_cleanned))

        return user_item_rating_tags

    def raw_folds(self):

        if self.shuffle:
            random.shuffle(self.user_item_rating_tags)
            self.shuffle = False  # set to false for future calls to raw_folds

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

    def folds(self):
        """Generator function to iterate over the folds of the Dataset.

        See :ref:`User Guide <iterate_over_folds>` for usage.

        Yields:
            tuple: :class:`Trainset` and testset of current fold.
        """

        for raw_trainset, raw_testset in self.raw_folds():
            trainset = self.construct_trainset(raw_trainset)
            testset = self.construct_testset(raw_testset)
            yield trainset, testset

    def construct_trainset(self, raw_trainset):

        trainset = Trainset(raw_trainset,
                            self.ratings_reader.rating_scale,
                            self.ratings_reader.offset,
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

    def cal_tag_freq(self):
        '''计算tag的出现次数

        '''
        tag_freq = defaultdict(int)
        for _, tags in self.raw_tags.items():
            for tag in tags:
                tag_freq[tag] += 1
        return tag_freq

    def info(self, diagram=False):
        '''计算每个tag的出现频数'''

        dataset = self.build_full_trainset()
        print("Total number of ratings: {}".format(dataset.n_ratings))
        print("Unique users: {}".format(dataset.n_users))
        print("Unique items: {}".format(dataset.n_items))
        print("Unique tags: {}".format(dataset.n_tags))
        print("Tag assignments: {}".format(dataset.tag_assignments))
        print("Average ratings per user: {}".format(
            dataset.n_ratings / dataset.n_users))
        print("Average tags per rating: {}".format(
            dataset.tag_assignments / dataset.n_ratings))

    def tag_power_law(self):
        ''' 统计标签的长尾分布

        '''
        fk = defaultdict(int)
        for key, value in self.tag_freq.items():
            fk[value] += 1

        plt.scatter(list(fk.keys()), np.array(
            list(fk.values())), c=np.random.rand(len(fk)))
        plt.title('标签流行度的长尾分布')
        plt.xlabel('流行度')  # 给 x 轴添加标签
        plt.ylabel('标签频度')  # 给 y 轴添加标签
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(0)
        plt.ylim(-1)
        plt.show()


class Trainset:
    """A trainset contains all useful data that constitutes a training set.

    It is used by the :meth:`train()
    <surprise.prediction_algorithms.algo_base.AlgoBase.train>` method of every
    prediction algorithm. You should not try to built such an object on your
    own but rather use the :meth:`Dataset.folds` method or the
    :meth:`DatasetAutoFolds.build_full_trainset` method.

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

    def cal_item_tag_freq(self):
        ''' cal the relevant tag occurrences for item.

        '''
        item_tag_freq = defaultdict(lambda: defaultdict(int))

        for _, iid, _, tids in self.uirts:
            for tid in tids:
                item_tag_freq[iid][tid] += 1

        # for iid, tags_freq in item_tag_freq.items():
        #     item_tag_freq[iid][-1] = sum([freq for tid, freq in tags_freq.items()])

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
