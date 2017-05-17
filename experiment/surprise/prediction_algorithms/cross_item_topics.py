from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from six.moves import range
from .algo_base import AlgoBase
from bidict import bidict
import lda


class CrossItemTopics(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, biased=True, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 n_topics=10, n_lda_iter=1000, alpha=0.01, eta=0.01,
                 verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.n_topics = n_topics
        self.n_lda_iter = n_lda_iter
        self.alpha = alpha
        self.eta = eta
        self.verbose = verbose

        AlgoBase.__init__(self)
        self.estimate_with_tags = True

    def train(self, trainset, aux_trainset):

        trainset.rank_sum_test(0.95)
        aux_trainset.rank_sum_test(0.95)

        trainset.construct()
        aux_trainset.construct()

        AlgoBase.train(self, trainset, aux_trainset)
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
        yt = np.zeros((self.n_topics,
                       self.n_factors), np.double)

        # 构建主题模型
        # 文档数为目标集与辅助集的物品总和
        X = np.zeros(
            (trainset.n_items + aux_trainset.n_items, self.n_tags), dtype=int)

        for _, iid, _, tids in trainset.uirts:
            for tid in tids:
                itid = self.to_inner_tid(trainset.to_raw_tag(tid))
                X[iid, itid] += 1
        for _, iid, _, tids in aux_trainset.uirts:
            for tid in tids:
                itid = self.to_inner_tid(aux_trainset.to_raw_tag(tid))
                X[iid + trainset.n_items, itid] += 1

        self.lda_model = lda.LDA(n_topics=self.n_topics, n_iter=self.n_lda_iter,
                                 alpha=self.alpha, eta=self.eta, refresh=2000)
        self.lda_model.fit(X)

        global_mean = trainset.global_mean
        aux_global_mean = aux_trainset.global_mean

        lr_all = self.lr_all
        reg_all = self.reg_all

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            for u, i, r in aux_trainset.all_ratings():

                topic_prop = self.lda_model.doc_topic_[i + self.trainset.n_items]
                sum_yt = np.dot(topic_prop, yt)

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

                yt += lr_all * (err * np.dot(topic_prop.reshape(self.n_topics, 1),
                                             aux_pu[u].reshape(1, self.n_factors)) - reg_all * yt)

            for u, i, r in trainset.all_ratings():

                topic_prop = self.lda_model.doc_topic_[i]
                sum_yt = np.dot(topic_prop, yt)

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

                yt += lr_all * (err * np.dot(topic_prop.reshape(self.n_topics, 1),
                                             pu[u].reshape(1, self.n_factors)) - reg_all * yt)
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

            item_topic_prop = self.lda_model.doc_topic_[i]
            sum_yt = np.dot(item_topic_prop, self.yt)
            est += np.dot((self.qi[i] + sum_yt), self.pu[u])

        return est
