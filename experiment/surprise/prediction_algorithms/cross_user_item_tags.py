
import numpy as np
from six.moves import range

from .algo_base import AlgoBase
from bidict import bidict


class CrossUserItemTags(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, biased=True, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.lr_all = lr_all
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_all = reg_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.verbose = verbose

        AlgoBase.__init__(self)
        self.estimate_with_tags = True

    def train(self, trainset, aux_trainset):

        AlgoBase.train(self, trainset)
        self.aux_trainset = aux_trainset
        self.construct_all_tags_set()
        self.sgd(trainset, aux_trainset)

    def construct_all_tags_set(self):
        '''目标域和辅助域的所有标签的并集

        '''
        self.all_tags_set = self.trainset.tags_set.union(self.aux_trainset.tags_set)

        # 两个域tag转换为inner tid
        raw2inner_id_all_tags = bidict()
        current_t_index = 0
        for tag in self.all_tags_set:
            if tag not in raw2inner_id_all_tags:
                raw2inner_id_all_tags[tag] = current_t_index
                current_t_index += 1

        self.raw2inner_id_all_tags = raw2inner_id_all_tags
        self.n_tags = len(raw2inner_id_all_tags)

    def to_inner_tid(self, rtag):
        try:
            return self.raw2inner_id_all_tags[rtag]
        except KeyError:
            raise ValueError(('raw tag id ' + str(rtag) +
                              ' is not part of the trainset.'))

    def to_raw_tag(self, itid):
        try:
            return self.raw2inner_id_all_tags.inv[itid]
        except KeyError:
            raise ValueError(('inner tag id ' + str(itid) +
                              ' is not part of the trainset.'))
    def knows_tag(self, tag):
        return tag in self.raw2inner_id_all_tags

    def sgd(self, trainset, aux_trainset):

        # 目标域的潜在向量
        # user biases
        bu = np.zeros(trainset.n_users, np.double)
        # item biases
        bi = np.zeros(trainset.n_items, np.double)
        # user factors
        pu = np.random.random((trainset.n_users, self.n_factors)
                              ) / np.sqrt(self.n_factors)
        # item factors
        qi = np.random.random((trainset.n_items, self.n_factors)
                              ) / np.sqrt(self.n_factors)

        # 辅助域的潜在向量
        # user biases
        aux_bu = np.zeros(aux_trainset.n_users, np.double)
        # item biases
        aux_bi = np.zeros(aux_trainset.n_items, np.double)
        # user factors
        aux_pu = np.random.random((aux_trainset.n_users, self.n_factors)
                                  ) / np.sqrt(self.n_factors)
        # item factors
        aux_qi = np.random.random((aux_trainset.n_items, self.n_factors)
                                  ) / np.sqrt(self.n_factors)

        # tag factors
        yt = np.zeros((self.n_tags,
                       self.n_factors), np.double)

        global_mean = trainset.global_mean if self.biased else 0
        aux_global_mean = aux_trainset.global_mean if self.biased else 0

        lr_all = self.lr_all
        reg_all = self.reg_all

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            # 辅助域sgd
            for u, i, r, tids in aux_trainset.all_ratings_tags():
                n_tags = max(len(tids), 1)
                sum_yt = np.sum([yt[self.to_inner_tid(aux_trainset.to_raw_tag(tid))]
                                 for tid in tids], axis=0) / n_tags
                # compute current error
                dot = np.dot((aux_qi[i] + sum_yt), aux_pu[u])
                err = r - (aux_global_mean + aux_bu[u] + aux_bi[i] + dot)
                # update biases
                if self.biased:
                    aux_bu[u] += lr_all * (err - reg_all * aux_bu[u])
                    aux_bi[i] += lr_all * (err - reg_all * aux_bi[i])
                # update factors
                aux_pu[u] += lr_all * \
                    (err * (aux_qi[i] + sum_yt) - reg_all * aux_pu[u])
                aux_qi[i] += lr_all * (err * aux_pu[u] - reg_all * aux_qi[i])
                for tid in tids:
                    t = self.to_inner_tid(aux_trainset.to_raw_tag(tid))
                    yt[t] += lr_all * (aux_pu[u] * (err / n_tags) - reg_all * yt[t])

            # 目标域sgd
            for u, i, r, tids in trainset.all_ratings_tags():
                n_tags = max(len(tids), 1)
                sum_yt = np.sum([yt[self.to_inner_tid(trainset.to_raw_tag(tid))]
                                 for tid in tids], axis=0) / n_tags
                # compute current error
                dot = np.dot((qi[i] + sum_yt), pu[u])
                err = r - (global_mean + bu[u] + bi[i] + dot)
                # update biases
                if self.biased:
                    bu[u] += lr_all * (err - reg_all * bu[u])
                    bi[i] += lr_all * (err - reg_all * bi[i])
                # update factors
                pu[u] += lr_all * (err * (qi[i] + sum_yt) - reg_all * pu[u])
                qi[i] += lr_all * (err * pu[u] - reg_all * qi[i])
                for tid in tids:
                    t = self.to_inner_tid(trainset.to_raw_tag(tid))
                    yt[t] += lr_all * (pu[u] * (err / n_tags) - reg_all * yt[t])

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yt = yt

    def estimate(self, u, i, tags):

        est = self.trainset.global_mean if self.biased else 0

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            yt_sum = np.zeros(self.n_factors, np.double)
            yt_cnt = 0

            for tag in tags:
                if self.knows_tag(tag):
                    tid = self.to_inner_tid(tag)
                    yt_sum += self.yt[tid]
                    yt_cnt += 1

            if yt_cnt != 0:
                yt_sum /= yt_cnt
            est += np.dot((self.qi[i] + yt_sum), self.pu[u])

        return est


class CrossUserItemRelTags(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, biased=True, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.lr_all = lr_all
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_all = reg_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.verbose = verbose

        AlgoBase.__init__(self)
        self.estimate_with_tags = True

    def train(self, trainset):

        trainset.rank_sum_test(confidence=0.95)
        trainset.construct()
        AlgoBase.train(self, trainset)
        self.sgd(trainset)

    def sgd(self, trainset):

        # user biases
        bu = np.zeros(trainset.n_users, np.double)
        # item biases
        bi = np.zeros(trainset.n_items, np.double)
        # user factors
        pu = np.random.random((trainset.n_users, self.n_factors)
                              ) / np.sqrt(self.n_factors)
        # item factors
        qi = np.random.random((trainset.n_items, self.n_factors)
                              ) / np.sqrt(self.n_factors)
        # tag factors
        yt = np.zeros((trainset.n_tags,
                       self.n_factors), np.double)

        lr_all = self.lr_all
        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi

        reg_all = self.reg_all
        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        global_mean = trainset.global_mean if self.biased else 0

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r, tids in trainset.all_ratings_tags():

                n_tags = max(len(tids), 1)
                sum_yt = np.sum([yt[tid] for tid in tids], axis=0) / n_tags

                # compute current error
                dot = np.dot((qi[i] + sum_yt), pu[u])
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                pu[u] += lr_pu * (err * (qi[i] + sum_yt) - reg_pu * pu[u])
                qi[i] += lr_qi * (err * pu[u] - reg_qi * qi[i])

                for t in tids:
                    yt[t] += lr_all * (pu[u] * (err / n_tags) - reg_all * yt[t])

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yt = yt

    def estimate(self, u, i, tags):

        est = self.trainset.global_mean if self.biased else 0

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            yt_sum = np.zeros(self.n_factors, np.double)
            yt_cnt = 0

            for tag in tags:
                if self.trainset.knows_tag(tag):
                    tid = self.trainset.to_inner_tid(tag)
                    yt_sum += self.yt[tid]
                    yt_cnt += 1

            if yt_cnt != 0:
                yt_sum /= yt_cnt
            est += np.dot((self.qi[i] + yt_sum), self.pu[u])

        return est
