import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from statistics import mean

def pre_processing(data):

	x = data.iloc[:, 3:13].values
	y = data.iloc[:, 13].values	

	label_encoder_x_1 = LabelEncoder()
	x[:, 1] = label_encoder_x_1.fit_transform(x[:, 1])
	
	label_encoder_x_2 = LabelEncoder()
	x[:, 2] = label_encoder_x_2.fit_transform(x[:, 2])

	onehotencoder = OneHotEncoder(categorical_features = [1])
	x = onehotencoder.fit_transform(x).toarray()

	x = x[:, 1:]

	return x, y


def scaling(x):
	sc = StandardScaler()
	x = sc.fit_transform(x)

	return x

if __name__ == '__main__':

	data = pd.read_csv('data.csv')

	x, target = pre_processing(data)

	x = scaling(x)

	NUM = 7000

	est = Ridge()
	est.fit(x[:NUM, :], target[:NUM])

	print('Before: ' + str(est.score(x[NUM:, :], target[NUM:])))

	function_set = ['add', 'sub', 'mul', 'div',
				'sqrt', 'log', 'abs', 'neg', 'inv',
				'max', 'min', 'sin', 'cos', 'tan']

	gp = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, n_jobs=3,
                         metric='spearman')
 
	gp.fit(x[:NUM, :], target[:NUM])


	gp_features = gp.transform(x)
	x = np.hstack((x, gp_features))


	est = Ridge()
	est.fit(x[:NUM, :], target[:NUM])
	print('After: ' + str(est.score(x[NUM:, :], target[NUM:])))

	params = {
		'objective':'binary', 
		'max_depth':5,
		'n_estimators':200,
		'learning_rate':0.1,
		'colsample_bytree':0.3,
		'num_leaves':8,
		'metric':'auc',
		'n_jobs':-1
	}

	folds = KFold(n_splits=5, shuffle=False, random_state=2019)

	print('Light GBM Model')

	acc = list()

	for fold_, (trn_idx, val_idx) in enumerate(folds.split(x, target)):
		print("Fold {}: ".format(fold_+1))
		clf = lgb.LGBMClassifier(**params)
		clf.fit(x[trn_idx], target[trn_idx], eval_set=[(x[val_idx], target[val_idx])], verbose=0, early_stopping_rounds=500)


		y_pred_test = clf.predict_proba(x[val_idx])
		y_pred_k = np.argmax(y_pred_test, axis=1)
		acc.append(accuracy_score(target[val_idx], y_pred_k))

	print(mean(acc))

	