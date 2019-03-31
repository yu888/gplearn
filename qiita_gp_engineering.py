# rng = check_random_state(0)
# boston = load_boston()
# perm = rng.permutation(boston.target.size)
# boston.data = boston.data[perm]
# boston.target = boston.target[perm]

# est = Ridge()
# est.fit(boston.data[:300, :], boston.target[:300])
# print est.score(boston.data[300:, :], boston.target[300:])

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor

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

	x, y = pre_processing(data)

	x = scaling(x)

	est = Ridge()
	est.fit(x[:300, :], y[:300])
	print(est.score(x[300:, :], y[300:]))

	function_set = ['add', 'sub', 'mul', 'div',
				'sqrt', 'log', 'abs', 'neg', 'inv',
				'max', 'min']

	gp = SymbolicRegressor(generations=20, population_size=2000,
						 hall_of_fame=100, n_components=10,
						 function_set=function_set,
						 parsimony_coefficient=0.0005,
						 max_samples=0.9, verbose=1,
						 random_state=0, n_jobs=3)

	gp.fit(x[:300, :], y[:300])


	gp_features = gp.transform(x)
	new_boston = np.hstack((x, gp_features))


	est = Ridge()
	est.fit(new_boston[:300, :], y[:300])
	print(est.score(new_boston[300:, :], y[300:]))


