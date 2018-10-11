'''
Author: Jing Y.
Date: Oct. 2018
'''

import math
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

class KMeans():
	def __init__(self, c, tol = 0.001, max_iter = 100):
		self.c = c  ## number of clusters
		self.tol = tol  ## tolerance
		self.max_iter = max_iter  ## max iteration

	def fit(self, data):
		print(data.shape)
		self.clusters = {}
		self.centroids = []
		## init cetroids 
		for i in range(self.c):
			self.centroids.append(data[i])
			self.clusters[i] = [ data[i] ]
		_iter = 0
		
		while _iter < self.max_iter:
			_iter += 1
			self._iteration(data) 
			## break condition that error is smaller than tolerance
			if self.err < self.tol:
				break
		
	def _iteration(self, data):
		self.err = 0
		self.clusters = {}
		## each iteration, grouping data into nearest cluster that having minimum distance to its centroid
		for each_data in data:
			each_data = list(each_data)
			c, eps = self.__assign_c(each_data, with_err = True)
			self.err += eps
			if c in self.clusters:
				self.clusters[c].append(each_data)
			else:
				self.clusters[c] = [each_data]

		## update centroids for each cluster
		for i in self.clusters.keys():
			self.centroids[i] = self.__get_centroid(self.clusters[i])

	def predict(self, new_data):
		re = []
		for each_data in new_data:
			each_data = list(each_data)
			c = self.__assign_c(each_data)
			re.append(c)

		return re

	def visualize(self):
		colors = ["r", "g", "c", "b", "k"]
		
		## can only predict first 2-dim data
		for i in range(len(self.centroids)):
			plt.scatter(self.centroids[i][0], self.centroids[i][1], s = 130, marker = "x", color = colors[i])
		
		for c in self.clusters.keys():
			for each in self.clusters[c]:
				plt.scatter(each[0], each[1], color = colors[c], s = 30)
		
		plt.show()

	def __assign_c(self, list_x, with_err = False):
		'''
		assign cluster to each data point
		'''
		distances = self._get_distance_batch(list_x, self.centroids)
		min_dis = sorted(range(len(distances)), key = distances.__getitem__)[0]
		if with_err:
			return min_dis, distances[min_dis]
		return min_dis

	def __get_centroid(self, list_x):
		return np.mean(np.array(list_x), axis = 0)
	
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
			except Exception as err:
				re.append(None)
		return re

if __name__ == '__main__':
	data = pd.read_csv("data/ipl.csv")
	from sklearn.utils import shuffle
	data = shuffle(data)
	data = np.array(data)
	model = KMeans(c = 3)

	model.fit(data)
	model.visualize()

	# print(model.predict(data))