"""
The :mod:`prediction_algorithms` package includes the prediction algorithms
available for recommendation.

The available prediction algorithms are:

.. autosummary::
    :nosignatures:

    random_pred.NormalPredictor
    baseline_only.BaselineOnly
    knns.KNNBasic
    knns.KNNWithMeans
    knns.KNNBaseline
    matrix_factorization.SVD
    matrix_factorization.SVDpp
    matrix_factorization.NMF
    slope_one.SlopeOne
    co_clustering.CoClustering
"""

from .algo_base import AlgoBase
from .random_pred import NormalPredictor
from .baseline_only import BaselineOnly
from .knns import KNNBasic
from .knns import KNNBaseline
from .knns import KNNWithMeans
from .knns import SymmetricAlgo
# from .matrix_factorization import SVD
# from .matrix_factorization import SVDpp
# from .matrix_factorization import NMF
# from .slope_one import SlopeOne
# from .co_clustering import CoClustering
from .user_item_tags import UserItemTags, UserItemGenomeTags, UserItemRelTags
from .item_rel_tags import ItemRelTags
from .item_topics import ItemTopics, ItemTopicsTest
from .user_item_topics import UserItemTopics
from .cross_user_item_tags import CrossUserItemTags, CrossUserItemRelTags
from .cross_item_rel_tags import CrossItemRelTags
from .cross_item_topics import CrossItemTopics, CrossItemTopicsTest
from .svd import SVD


from .predictions import PredictionImpossible
from .predictions import Prediction

__all__ = ['AlgoBase', 'NormalPredictor', 'BaselineOnly', 'KNNBasic',
           'KNNBaseline', 'KNNWithMeans', 'SVD', 'SVDpp', 'NMF', 'SlopeOne',
           'CoClustering', 'PredictionImpossible', 'Prediction', 'SymmetricAlgo',
           'UserItemTags', 'UserItemGenomeTags', 'ItemRelTags', 'UserItemRelTags', 'ItemTopics', 'UserItemTopics',
           'CrossUserItemTags', 'CrossUserItemRelTags', 'CrossItemRelTags', 'CrossItemTopics',
           'ItemTopicsTest', 'CrossItemTopicsTest']
