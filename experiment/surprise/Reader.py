import pandas as pd
from collections import defaultdict


class LTReader():
    """Abstract class where is used to parse a file containing ratings or tags.

    Such a file is assumed to specify only one value per line, and each line
    needs to respect the following structure: ::

        user ; item ; value ; [timestamp]

    where the order of the fields and the seperator (here ';') may be
    arbitrarily defined (see below).  brackets indicate that the timestamp
    field is optional.

    """

    def __init__(self, txt_file=None, sep=None, skip_lines=0):

        self.txt_file = txt_file
        self.sep = sep
        self.skip_lines = skip_lines

    def read(self):
        """Return a dict of user,item,value"""

        raw_ratings = list()
        tag_dict = defaultdict(list)
        last_rating = None
        with open(self.txt_file, 'r') as f:
            for line in f.readlines():
                split_line = line.split(' ')
                uid = split_line[0]
                iid = split_line[1]
                rating = float(split_line[2]) * 0.5
                tag = ' '.join([t.strip().lower() for t in split_line[3:]])
                if (uid, iid, rating) != last_rating:
                    raw_ratings.append((uid, iid, rating))
                    last_rating = (uid, iid, rating)
                tag_dict[(uid, iid)].append(tag)

        print("read {} lines from {}.".format(len(raw_ratings), self.txt_file))
        return raw_ratings, tag_dict


class Reader():
    """Abstract class where is used to parse a file containing ratings or tags.

    Such a file is assumed to specify only one value per line, and each line
    needs to respect the following structure: ::

        user ; item ; value ; [timestamp]

    where the order of the fields and the seperator (here ';') may be
    arbitrarily defined (see below).  brackets indicate that the timestamp
    field is optional.

    """

    def __init__(self, csv_file=None, sep=None, skip_lines=0, first_n_ratings=None):

        # self.reader = csv.reader(open(csv_file))
        # df = pd.read_csv('ml-100k/u.data', sep='\t',)
        self.csv_file = csv_file
        self.sep = sep
        self.skip_lines = skip_lines
        self.first_n_ratings = first_n_ratings

    def read(self):
        """Return a dict of user,item,value"""
        # raw_lines = list()
        # for line in pd.read_csv(self.csv_file, sep=self.sep,
        # skiprows=self.skip_lines, encoding='utf-8').itertuples():
        raw_lines = list(pd.read_csv(self.csv_file, sep=self.sep,
                                     skiprows=self.skip_lines, encoding='utf-8').itertuples())
        print("read {} lines from {}.".format(len(raw_lines), self.csv_file))
        return raw_lines


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

    def __init__(self, dataset_path=None, sep=None, rating_scale=(1, 5), skip_lines=0):

        Reader.__init__(self, dataset_path + 'ratings.csv', sep, skip_lines)
        self.rating_scale = rating_scale
        lower_bound, higher_bound = rating_scale
        # offset的计算方法存在疑问
        self.offset = -lower_bound + 1 if lower_bound <= 0 else 0

    def read(self):
        """Return a list of user,item,rating"""
        raw_ratings = list()
        for i, uid, iid, rating, _ in super().read():
            raw_ratings.append((uid, iid, float(rating)))
        return raw_ratings


class TagsReader(Reader):
    """The TagsReader class is used to parse a file containing tags.

    Args:
        line_format(:obj:`string`): The fields names, in the order at which
            they are encountered on a line. Example: ``'item user tag'``.
        sep(char): the separator between fields. Example : ``';'``.
        skip_lines(:obj:`int`, optional): Number of lines to skip at the
            beginning of the file. Default is ``0``.

    """

    def __init__(self, dataset_path=None, sep=None, skip_lines=0):
        Reader.__init__(self, dataset_path + 'tags.csv', sep, skip_lines)
        self.dataset_path = dataset_path

    def read(self):
        """Return a dict of user,item,value"""

        tag_dict = defaultdict(list)
        for i, uid, iid, tags, _ in super().read():
            tags = str(tags)
            tags = [t.strip().lower() for t in tags.split(',')]
            tag_dict[(uid, iid)].extend(tags)

        return tag_dict

    def read_genome(self):

        # 读取genome_tags的id
        genome_tags_id = dict()
        for _, tid, tag in pd.read_csv(self.dataset_path + 'genome-tags.csv',
                                       sep=self.sep, skiprows=self.skip_lines).itertuples():
            genome_tags_id[tag] = tid

        # 读取genome-tag与item的相关性
        genome_score = dict()
        for _, iid, tid, rel_score in pd.read_csv(self.dataset_path + 'genome-scores.csv', sep=self.sep,
                                                  skiprows=self.skip_lines).itertuples():
            genome_score[(iid, tid)] = rel_score

        return genome_tags_id, genome_score
