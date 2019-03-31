import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.datasets import load_boston
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness
from statistics import mean
from math import sqrt

if __name__ == '__main__':

	rng = check_random_state(0)
	boston = load_boston()
	perm = rng.permutation(boston.target.size)
	boston.data = boston.data[perm]
	boston.target = boston.target[perm]

	function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min']

	gp = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, n_jobs=3)

	gp.fit(boston.data[:300, :], boston.target[:300])

	gp_features = gp.transform(boston.data)
	boston.data = np.hstack((boston.data, gp_features))

	print(gp._best_programs[0])

	est = Ridge()
	est.fit(new_boston[:300, :], boston.target[:300])
	print(est.score(new_boston[300:, :], boston.target[300:]))
    print(gp._best_programs[0])

	params = {
		'objective':'regression', 
		'max_depth':-1,
		'n_estimators':100,
		'learning_rate':0.1,
		'colsample_bytree':0.3,
		'num_leaves':8,
		'metric':'auc',
		'n_jobs':-1
	}

	folds = KFold(n_splits=5, shuffle=False, random_state=2019)

	print('Light GBM Model')

	acc = list()

	for fold_, (trn_idx, val_idx) in enumerate(folds.split(boston.data, boston.target)):
		print("Fold {}: ".format(fold_+1))
		reg = lgb.LGBMRegressor(**params)
		reg.fit(boston.data[trn_idx], boston.target[trn_idx], eval_set=[(boston.data[val_idx], boston.target[val_idx])], verbose=0, early_stopping_rounds=500)


		y_pred = reg.predict(boston.data[val_idx])
		acc.append(sqrt(mean_squared_error(boston.target[val_idx], y_pred) ** 0.5))


	print(mean(acc))

