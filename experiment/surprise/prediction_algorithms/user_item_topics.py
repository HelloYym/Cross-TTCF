from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from six.moves import range
import lda
from .algo_base import AlgoBase


class UserItemTopics(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, biased=True, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 n_topics=20, n_lda_iter=2000, alpha=0.1, eta=0.01,
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
        self.n_topics = n_topics
        self.n_lda_iter = n_lda_iter
        self.alpha = alpha
        self.eta = eta
        self.verbose = verbose

        AlgoBase.__init__(self)
        self.estimate_with_tags = True

    def train(self, trainset):

        AlgoBase.train(self, trainset)
        self.sgd(trainset)

    def sgd(self, trainset):

        n_users = trainset.n_users
        n_items = trainset.n_items
        n_tags = trainset.n_tags
        n_topics = self.n_topics

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

        vocab = [trainset.to_raw_tag(tid) for tid in range(n_tags)]
        self.lda_model = lda.LDA(n_topics=n_topics, n_iter=self.n_lda_iter,
                                 alpha=self.alpha, eta=self.eta, refresh=2000)
        self.lda_model.fit(X)

        # topic info
        topic_word = self.lda_model.topic_word_
        n_top_words = 10
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
            print('Topic {}: {}'.format(i, '; '.join(topic_words)))

        topic_prop = list()
        for u, i, r, tids in trainset.all_ratings_tags():
            X = np.zeros((1, n_tags), dtype=int)
            for tid in tids:
                X[0, tid] += 1

            # 最大的主题
            # topic_prop.append(self.lda_model.transform(X)[0].argmax())
            topic_prop.append(self.lda_model.transform(X)[0])

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            for tp, (u, i, r, tids) in zip(topic_prop, trainset.all_ratings_tags()):

                # compute current error
                sum_yt = np.dot(tp, yt)
                dot = np.dot((qi[i] + sum_yt), pu[u])
                # dot = np.dot((qi[i] + yt[t]), pu[u])
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                pu[u] += lr_pu * (err * (qi[i] + sum_yt) - reg_pu * pu[u])
                qi[i] += lr_qi * (err * pu[u] - reg_qi * qi[i])

                # yt[t] += lr_all * (pu[u] * err - reg_all * yt[t])

                yt += lr_all * (err * np.dot(tp.reshape(n_topics, 1),
                                             pu[u].reshape(1, self.n_factors)) - reg_all * yt)

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

            X = np.zeros((1, self.trainset.n_tags), dtype=int)

            for tag in tags:
                if self.trainset.knows_tag(tag):
                    tid = self.trainset.to_inner_tid(tag)
                    X[0, tid] += 1
            if np.sum(X) > 0:
                topic_prop = self.lda_model.transform(X)[0]
                # t = topic_prop.argmax()
                sum_yt = np.dot(topic_prop, self.yt)
                # sum_yt = self.yt[t]
            else:
                sum_yt = 0

            est += np.dot((self.qi[i] + sum_yt), self.pu[u])

        return est
