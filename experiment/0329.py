from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise.evaluate import GridSearch
from surprise import Reader
import os


# path to dataset file
file_path = os.path.expanduser('ml-latest-small/ratings.csv')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating timestamp', rating_scale=(0.5, 5),
                skip_lines=1, sep=',')

data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD()

param_grid = {'n_epochs': [5, 10, 50]}

grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])

grid_search.evaluate(data)

# best RMSE score
print(grid_search.best_score['RMSE'])
# >>> 0.96117566386

# combination of parameters that gave the best RMSE score
print(grid_search.best_params['RMSE'])
# >>> {'reg_all': 0.4, 'lr_all': 0.005, 'n_epochs': 10}
