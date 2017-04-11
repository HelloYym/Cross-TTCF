"""
the :mod:`dataset` module defines some tools for managing datasets.

Summary:

.. autosummary::
    :nosignatures:

    Dataset.load_builtin
    Dataset.load_from_file
    Dataset.load_from_folds
    Dataset.folds
    DatasetAutoFolds.split
    Reader
    Trainset
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
from collections import namedtuple
import sys
import os
import zipfile
import itertools
import random
import shlex

import numpy as np
from six.moves import input
from six.moves import range
from six import iteritems


# directory where builtin datasets are stored. For now it's in the home
# directory under the .surprise_data. May be ask user to define it?
DATASETS_DIR = os.path.expanduser('~') + '/.surprise_data/'


class Dataset:
    """Base class for loading datasets.

    Note that you should never instantiate the :class:`Dataset` class directly
    (same goes for its derived classes), but instead use one of the three
    available methods for loading datasets."""

    def __init__(self, ratings_file=None, tags_file=None, ratings_reader=None, tags_reader=None):

        self.ratings_reader = ratings_reader
        self.tags_reader = tags_reader
        self.ratings_file = ratings_file
        self.tags_file = tags_file
        self.n_folds = 5
        self.shuffle = True
        self.raw_ratings = self.read_ratings(self.ratings_file)
        self.raw_tags = self.read_tags(self.tags_file)

    def read_ratings(self, file_name):
        """Return a list of ratings (user, item, rating, timestamp) read from
        file_name"""

        with open(os.path.expanduser(file_name)) as f:
            raw_ratings = [self.ratings_reader.parse_line(line) for line in
                           itertools.islice(f, self.ratings_reader.skip_lines, None)]
        return raw_ratings

    def read_tags(self, file_name):
        """Return a list of tags (user, item, tags, timestamp) read from
        file_name"""

        # 将(user,item)的多个标签聚集在一起
        with open(os.path.expanduser(file_name)) as f:
            raw_tags = []
            last_uid = last_iid = None
            tags = []

            for line in itertools.islice(f, self.tags_reader.skip_lines, None):
                uid, iid, tag, timestamp = self.tags_reader.parse_line(line)
                tag = tag.split(',')
                if uid == last_uid and iid == last_iid or len(tags) == 0:
                    tags.extend([t.strip() for t in tag])
                else:
                    raw_tags.append((uid, iid, tags))
                    tags = [t.strip() for t in tag]

        return raw_tags

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

        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
        for urid, irid, r, timestamp in raw_trainset:
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

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        n_users = len(ur)  # number of users
        n_items = len(ir)  # number of items
        n_ratings = len(raw_trainset)

        trainset = Trainset(ur,
                            ir,
                            n_users,
                            n_items,
                            n_ratings,
                            self.ratings_reader.rating_scale,
                            self.ratings_reader.offset,
                            raw2inner_id_users,
                            raw2inner_id_items)

        return trainset

    def construct_testset(self, raw_testset):

        return [(ruid, riid, r_ui_trans)
                for (ruid, riid, r_ui_trans, _) in raw_testset]

    def build_full_trainset(self):
        """Do not split the dataset into folds and just return a trainset as
        is, built from the whole dataset.

        """

        return self.construct_trainset(self.raw_ratings)

    def raw_folds(self):

        if self.shuffle:
            random.shuffle(self.raw_ratings)
            random.shuffle(self.raw_tags)
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

        return k_folds(self.raw_ratings, self.n_folds)

    def split(self, n_folds=5, shuffle=True):
        """Split the dataset into folds for futur cross-validation.

        If you forget to call :meth:`split`, the dataset will be automatically
        shuffled and split for 5-folds cross-validation.

        You can obtain repeatable splits over your all your experiments by
        seeding the RNG: ::

            import random
            random.seed(my_seed)  # call this before you call split!

        Args:
            n_folds(:obj:`int`): The number of folds.
            shuffle(:obj:`bool`): Whether to shuffle ratings before splitting.
                If ``False``, folds will always be the same each time the
                experiment is run. Default is ``True``.
        """

        self.n_folds = n_folds
        self.shuffle = shuffle


class Reader():
    """Abstract class where is used to parse a file containing ratings or tags.

    Such a file is assumed to specify only one value per line, and each line
    needs to respect the following structure: ::

        user ; item ; value ; [timestamp]

    where the order of the fields and the seperator (here ';') may be
    arbitrarily defined (see below).  brackets indicate that the timestamp
    field is optional.

    """

    def __init__(self, line_format=None, sep=None, skip_lines=0):

        self.sep = sep
        self.skip_lines = skip_lines

        self.entities = line_format.split()

        self.with_timestamp = True if 'timestamp' in self.entities else False

    def parse_line(self, line):
        '''Parse a line.

        Args:
            line(str): The line to parse

        Returns:
            tuple: User id, item id, value and timestamp. The timestamp is set
            to ``None`` if it does no exist.
            '''

        line = line.replace('\'', '')
        line = shlex.shlex(line, posix=True)
        line.whitespace = self.sep
        line.whitespace_split = True
        line = list(line)

        try:
            if self.with_timestamp:
                uid, iid, value, timestamp = (k.strip() for k in line)
            else:
                uid, iid, value = (k.strip() for k in line)
                timestamp = None

        except IndexError:
            raise ValueError(('Impossible to parse line.' +
                              ' Check the line_format  and sep parameters.'))

        return uid, iid, value, timestamp


class RatingsReader(Reader):
    """The RatingsReader class is used to parse a file containing ratings.

    Args:
        line_format(:obj:`string`): The fields names, in the order at which
            they are encountered on a line. Example: ``'item user rating'``.
        sep(char): the separator between fields. Example : ``';'``.
        rating_scale(:obj:`tuple`, optional): The rating scale used for every
            rating.  Default is ``(1, 5)``.
        skip_lines(:obj:`int`, optional): Number of lines to skip at the
            beginning of the file. Default is ``0``.

    """

    def __init__(self, line_format=None, sep=None,
                 rating_scale=(1, 5), skip_lines=0):

        Reader.__init__(self, line_format, sep, skip_lines)
        self.rating_scale = rating_scale
        lower_bound, higher_bound = rating_scale
        self.offset = -lower_bound + 1 if lower_bound <= 0 else 0

    def parse_line(self, line):
        '''Parse a line.

        Ratings are translated so that they are all strictly positive.

        Args:
            line(str): The line to parse

        Returns:
            tuple: User id, item id, rating and timestamp. The timestamp is set
            to ``None`` if it does no exist.
            '''

        uid, iid, value, timestamp = super().parse_line(line)
        rating = float(value)
        rating += self.offset

        return uid, iid, rating, timestamp


class TagsReader(Reader):
    """The TagsReader class is used to parse a file containing tags.

    Args:
        line_format(:obj:`string`): The fields names, in the order at which
            they are encountered on a line. Example: ``'item user tag'``.
        sep(char): the separator between fields. Example : ``';'``.
        skip_lines(:obj:`int`, optional): Number of lines to skip at the
            beginning of the file. Default is ``0``.

    """

    def __init__(self, line_format=None, sep=None, skip_lines=0):

        Reader.__init__(self, line_format, sep, skip_lines)

    def parse_line(self, line):
        '''Parse a line.

        Args:
            line(str): The line to parse

        Returns:
            tuple: User id, item id, tag and timestamp. The timestamp is set
            to ``None`` if it does no exist.
            '''

        uid, iid, tag, timestamp = super().parse_line(line)

        return uid, iid, tag, timestamp


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
        n_ratings: Total number of ratings :math:`|R_{train}|`.
        rating_scale(tuple): The minimum and maximal rating of the rating
            scale.
        global_mean: The mean of all ratings :math:`\\mu`.
    """

    def __init__(self, ur, ir, n_users, n_items, n_ratings, rating_scale,
                 offset, raw2inner_id_users, raw2inner_id_items):

        self.ur = ur
        self.ir = ir
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self.rating_scale = rating_scale
        self.offset = offset
        self._raw2inner_id_users = raw2inner_id_users
        self._raw2inner_id_items = raw2inner_id_items
        self._global_mean = None

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

    def to_inner_uid(self, ruid):
        """Convert a raw **user** id to an inner id.

        See :ref:`this note<raw_inner_note>`.

        Args:
            ruid(str): The user raw id.

        Returns:
            int: The user inner id.

        Raises:
            ValueError: When user is not part of the trainset.
        """

        try:
            return self._raw2inner_id_users[ruid]
        except KeyError:
            raise ValueError(('User ' + str(ruid) +
                              ' is not part of the trainset.'))

    def to_inner_iid(self, riid):
        """Convert a raw **item** id to an inner id.

        See :ref:`this note<raw_inner_note>`.

        Args:
            riid(str): The item raw id.

        Returns:
            int: The item inner id.

        Raises:
            ValueError: When item is not part of the trainset.
        """

        try:
            return self._raw2inner_id_items[riid]
        except KeyError:
            raise ValueError(('Item ' + str(riid) +
                              ' is not part of the trainset.'))

    def all_ratings(self):
        """Generator function to iterate over all ratings.

        Yields:
            A tuple ``(uid, iid, rating)`` where ids are inner ids.
        """

        for u, u_ratings in iteritems(self.ur):
            for i, r in u_ratings:
                yield u, i, r

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
