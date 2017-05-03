
from collections import defaultdict
import random
import numpy as np
import matplotlib.pyplot as plt

from six.moves import range

from .Reader import RatingsReader, TagsReader
from .trainset import Trainset


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
        # self.rank_sum_test(confidence=1.0)
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

    # def rank_sum_test(self, confidence=0.95):
    #     # 具有某个tag的rating数肯定不会超过50%

    #     # 先计算每一个分数的秩
    #     # 每个rating的个数
    #     ratings_num = defaultdict(int)
    #     for _, _, r, tags in self.user_item_rating_tags:
    #         ratings_num[r] += 1
    #     # 按中位数作为秩
    #     ratings_median = defaultdict(int)
    #     c = 0
    #     for r in np.arange(0.5, 5.5, 0.5):
    #         ratings_median[r] = c + (ratings_num[r] + 1) * 0.5
    #         c += ratings_num[r]

    #     # 每个tag的秩list
    #     tag_ranks_dict = defaultdict(list)
    #     for _, _, r, tags in self.user_item_rating_tags:
    #         ratings_num[r] += 1
    #         for tag in tags:
    #             tag_ranks_dict[tag].append(ratings_median[r])

    #     all_size = len(self.user_item_rating_tags)
    #     for tag, tag_ranks in tag_ranks_dict.items():
    #         m = len(tag_ranks)
    #         n = all_size - m
    #         # tag的秩和
    #         rank_sum = sum(tag_ranks)
    #         # 进行rank-sum test
    #         halfMsum = 0.5 * m * (m + n + 1)
    #         twelthMNsum = (1.0 / 6) * halfMsum * n
    #         zNumerator = rank_sum - halfMsum
    #         zDenominator = twelthMNsum ** 0.5
    #         z = abs(zNumerator / zDenominator)
    #         # 如果结果小于置信度，说明该tag对评分影响不大，删除该评分
    #         if z < confidence:
    #             self.tags_set.discard(tag)

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
        dataset.info()

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
