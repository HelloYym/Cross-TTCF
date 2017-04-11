"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import AlgoBase
from surprise import SymmetricAlgo
from surprise import PredictionImpossible


class KNNBasicTest(SymmetricAlgo):
    """A basic collaborative filtering algorithm.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \\frac{
        \\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v) \cdot r_{vi}}
        {\\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v)}

    or

    .. math::
        \hat{r}_{ui} = \\frac{
        \\sum\\limits_{j \in N^k_u(i)} \\text{sim}(i, j) \cdot r_{uj}}
        {\\sum\\limits_{j \in N^k_u(j)} \\text{sim}(i, j)}

    depending on the ``user_based`` field of the ``sim_options`` parameter.

    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation (see :ref:`this note <actual_k_note>`). Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set the the global mean of all ratings. Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k

    def train(self, trainset):

        SymmetricAlgo.train(self, trainset)
        self.sim = self.compute_similarities()

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        # 只选出了那些用户u关联的item
        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (_, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details










