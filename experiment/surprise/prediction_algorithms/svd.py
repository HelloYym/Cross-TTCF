import numpy as np
from six.moves import range

from .algo_base import AlgoBase
from .predictions import PredictionImpossible


class SVD(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, biased=True, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.verbose = verbose

        AlgoBase.__init__(self)

    def train(self, trainset):

        AlgoBase.train(self, trainset)
        self.sgd(trainset)

    def sgd(self, trainset):

        # user biases
        bu = np.zeros(trainset.n_users, np.double)
        # item biases
        bi = np.zeros(trainset.n_items, np.double)
        # user factors
        pu = np.random.random(
            (trainset.n_users, self.n_factors)) / np.sqrt(self.n_factors)
        # item factors
        qi = np.random.random(
            (trainset.n_items, self.n_factors)) / np.sqrt(self.n_factors)

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        global_mean = self.trainset.global_mean

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                # compute current error
                dot = np.dot(pu[u], qi[i])
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                pu[u] += lr_pu * (err * qi[i] - reg_pu * pu[u])
                qi[i] += lr_qi * (err * pu[u] - reg_qi * qi[i])

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):
        # Should we cythonize this as well?

        est = self.trainset.global_mean

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            est += np.dot(self.qi[i], self.pu[u])

        return est
