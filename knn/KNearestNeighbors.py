'''
Author: Jing Y.
Date: Oct, 2018
'''

import pandas as pd 
import math
import numpy as np

class KNearestNeighbours():
	'''
	Simple Implementation of KNN model
	
	'''
	def __init__(self, k):
		'''
		Hyperparameter:
			- k: top K neighbors selected for voting
		'''
		self.k = k

	def fit(self, x, y):
		'''
		to set up training set to select neighbors
		'''
		self.train_x = x
		self.train_y = y

	def predict_x(self, x_new):
		'''
		to predict each instance
		'''
		distance_list = self._get_distance_batch(x_new, self.train_x)
		neighbors_ind = sorted(range(len(distance_list)), key = distance_list.__getitem__)[:self.k]
		neighbors = []
		for ind in neighbors_ind:
			neighbors.append(self.train_y[ind])
		
		y_pred = self.__get_majority_vote(neighbors)
		
		return y_pred
	
	def predict(self, x, y_true):
		'''
		to predict batch of instances
		'''
		y_pred = []
		for xi in x:
			xi = list(xi)
			y_pred.append(self.predict_x(xi))

		score = self.__get_score(y_true, y_pred)
		print("accuracy: ", round(score,4) )

	def _get_distance_batch(self, x1, list_x2):
		'''
		to get distances between x1 and list of x
		for parallel speed up
		'''
		re = []
		for x2 in list_x2:
			try:
				x2 = list(x2)
				re.append(self.__get_distance(x1, x2))
			except:
				re.append(None)
		return re

	def __get_majority_vote(self, neighbors):
		'''
		to get majority voting result among neighbors
		'''
		votes = {}
		for v in neighbors:
			if v not in votes:
				votes[v] = 1
			else:
				votes[v] += 1
		re = sorted(votes.items(), key = lambda items: items[1])[0][0]
		return re
	
	def __get_distance(self, x1, x2):
		'''
		x1, x2 are 1-dim vectors like [1, 6, 5, ... 3]
		'''
		assert type(x1) is list
		assert type(x2) is list
		assert len(x1) == len(x2)
		
		ss = 0
		for i in range(len(x1)):
			ss += pow((x1[i] - x2[i]), 2)
		
		return math.sqrt(ss) 

	def __get_score(self, y_true, y_pred):
		'''
		to get accuracy score
		'''
		s = [1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))]
		
		return sum(s) / len(s)


if __name__ == '__main__':
	data = pd.read_csv("data/iris.data.csv")
	from sklearn.utils import shuffle
	data = shuffle(data)
	n = data.shape[0]
	split = int(n*0.6)
	train_x = np.array(data.iloc[:split, :-1])
	train_y = np.array(data.iloc[:split, -1])
	test_x = np.array(data.iloc[split:, :-1])
	test_y = np.array(data.iloc[split:, -1])

	model = KNearestNeighbours(3)
	model.fit(train_x, train_y)
	model.predict(test_x, test_y)

