from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from six.moves import range
import copy
from .algo_base import AlgoBase
import lda


class ItemTopics(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, biased=True, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 confidence=0.95,
                 n_topics=10, n_lda_iter=1000, alpha=0.01, eta=0.01,
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
        self.n_topics = n_topics
        self.n_lda_iter = n_lda_iter
        self.alpha = alpha
        self.eta = eta
        self.verbose = verbose

        AlgoBase.__init__(self)
        self.estimate_with_tags = True

    def train(self, trainset):

        trainset.rank_sum_test(confidence=self.confidence)
        trainset.construct()
        AlgoBase.train(self, trainset)
        self.sgd(trainset)

    def sgd(self, trainset):

        n_users = trainset.n_users
        n_items = trainset.n_items
        n_tags = trainset.n_tags
        n_topics = self.n_topics

        # user biases
        bu = np.zeros(n_users, np.double)
        # item biases
        bi = np.zeros(n_items, np.double)

        # user factors
        pu = np.random.random((n_users, self.n_factors)
                              ) / np.sqrt(self.n_factors)
        # item factors
        qi = np.random.random((n_items, self.n_factors)
                              ) / np.sqrt(self.n_factors)
        # topic factors
        yt = np.random.random((n_topics, self.n_factors)
                              ) / np.sqrt(self.n_factors)

        # 考虑改为稀疏矩阵
        X = np.zeros((n_items, n_tags), dtype=int)
        for _, iid, _, tids in trainset.uirts:
            for tid in tids:
                X[iid, tid] += 1

        # vocab = [trainset.to_raw_tag(tid) for tid in range(n_tags)]
        self.lda_model = lda.LDA(n_topics=n_topics, n_iter=self.n_lda_iter,
                                 alpha=self.alpha, eta=self.eta, refresh=1000)
        self.lda_model.fit(X)

        # topic info
        # topic_word = self.lda_model.topic_word_
        # n_top_words = 10
        # for i, topic_dist in enumerate(topic_word):
        #     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        #     print('Topic {}: {}'.format(i, ' '.join(topic_words)))

        # self.item_topic = np.argmax(self.lda_model.doc_topic_, axis=1)

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

                # t = self.item_topic[i]
                topic_prop = self.lda_model.doc_topic_[i]
                sum_yt = np.dot(topic_prop, yt)

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

                yt += lr_all * (err * np.dot(topic_prop.reshape(n_topics, 1),
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


class ItemTopicsTest(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, biased=True, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 confidence=0.95,
                 n_topics=10, n_lda_iter=1000, alpha=0.01, eta=0.01,
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
        self.n_topics = n_topics
        self.n_lda_iter = n_lda_iter
        self.alpha = alpha
        self.eta = eta
        self.verbose = verbose

        AlgoBase.__init__(self)
        self.estimate_with_tags = True

    def train(self, trainset):

        trainset.rank_sum_test(confidence=self.confidence)
        trainset.construct()
        AlgoBase.train(self, trainset)
        self.sgd(trainset)

    def sgd(self, trainset):

        n_users = trainset.n_users
        n_items = trainset.n_items
        n_tags = trainset.n_tags
        n_topics = self.n_factors

        # user biases
        bu = np.zeros(n_users, np.double)
        # item biases
        bi = np.zeros(n_items, np.double)

        # user factors
        pu = np.random.random((n_users, self.n_factors)
                              ) / np.sqrt(self.n_factors)
        # item factors
        qi = np.random.random((n_items, self.n_factors)
                              ) / np.sqrt(self.n_factors)
        # topic factors
        yt = np.random.random((n_topics, self.n_factors)
                              ) / np.sqrt(self.n_factors)

        # 考虑改为稀疏矩阵
        X = np.zeros((n_items, n_tags), dtype=int)
        for _, iid, _, tids in trainset.uirts:
            for tid in tids:
                X[iid, tid] += 1

        # vocab = [trainset.to_raw_tag(tid) for tid in range(n_tags)]
        self.lda_model = lda.LDA(n_topics=n_topics, n_iter=self.n_lda_iter,
                                 alpha=self.alpha, eta=self.eta, refresh=2000)
        self.lda_model.fit(X)

        lr_all = self.lr_all
        reg_all = self.reg_all

        global_mean = trainset.global_mean

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                topic_prop = self.lda_model.doc_topic_[i]

                # compute current error
                dot = np.dot((qi[i] + topic_prop), pu[u])
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if self.biased:
                    bu[u] += lr_all * (err - reg_all * bu[u])
                    bi[i] += lr_all * (err - reg_all * bi[i])

                # update factors
                pu[u] += lr_all * \
                    (err * (qi[i] + topic_prop) - reg_all * pu[u])
                qi[i] += lr_all * (err * pu[u] - reg_all * qi[i])

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
            est += np.dot((self.qi[i] + item_topic_prop), self.pu[u])

        return est
