from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from six.moves import range
import copy
from .algo_base import AlgoBase


class ItemRelTags(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, biased=True, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 confidence = 0.95,
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
        self.confidence = confidence
        self.verbose = verbose

        AlgoBase.__init__(self)
        self.estimate_with_tags = True

    def train(self, trainset):

        trainset.rank_sum_test(confidence=self.confidence)
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

        global_mean = trainset.global_mean

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                item_tags = trainset.get_item_tags(i)

                n_tags = max(1, sum(item_tags.values()))
                sum_yt = np.sum(
                    [yt[tid] * freq for tid, freq in item_tags.items()], axis=0) / n_tags

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

                for t, freq in item_tags.items():
                    yt[t] += lr_all * \
                        (pu[u] * freq * (err / n_tags) - reg_all * yt[t])

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yt = yt

    def estimate(self, u, i, tags):

        est = self.trainset.global_mean

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):

            item_tags = copy.deepcopy(self.trainset.get_item_tags(i))

            yt_cnt = max(sum(item_tags.values()), 1)
            yt_sum = np.sum([self.yt[tid] * freq for tid,
                             freq in item_tags.items()], axis=0) / yt_cnt

            est += np.dot((self.qi[i] + yt_sum), self.pu[u])

        return est
