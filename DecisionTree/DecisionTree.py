'''
Author: Jing Y.
Date: Oct, 2018
'''
import pandas as pd 
import numpy as np 
import pprint

class DecisionTree():
	'''
	to implement CART tree model
	use gini index for best split criteria
	params:
		- max_depth: max depth of tree
		- min_size: min size in tree patition
	'''
	def __init__(self, max_depth = 3, min_size = 10):
		self.max_depth = max_depth  ## stop split if tree depth reaches max_depth
		self.min_size = min_size  ## stop split if patitions size smaller than min_size
		self.tree = None

	def fit(self, dataset):
		root = self.__find_best_split(dataset)
		self.__build_tree(root)
		self.tree = root

	def predict(self, dataset):
		if not self.tree:
			print("tree is empty or not yet built")
			return
		y = []
		for row in dataset:
			re = self._predcit_row(row, root = self.tree)
			y.append(re)
		score = self._eval(y_true = dataset[:, -1], y_est = y)
		print("accuracy score:", score)
		return y

	def _predcit_row(self, row, root):
		''' to predict each row, traverse the whole tree '''
		if row[root['attribute']] < root['threshold']:
			if type(root["L"]) is not dict:
				return root["L"]
			else:
				return self._predcit_row(row, root["L"])  ## dont forget return
		else:
			if type(root["R"]) is not dict:
				return root["R"]
			else:
				return self._predcit_row(row, root["R"])

	def _eval(self, y_true, y_est):
		''' evaluate y_true vs y_est '''
		assert len(y_true) == len(y_est)
		return sum(y_true == y_est) / len(y_true)

	def __build_tree(self, root, depth = 0):
		''' recursive call '''
		left, right = root["patitions"]
		del(root["patitions"])  ## no need to save patitions after split
		
		if len(left) == 0 or len(right) == 0:  ## one partition is empty, then merge
			child = left + right
			root["L"] = self.__get_vote(child)
			root["R"] = self.__get_vote(child)
			return

		if depth >= self.max_depth:  ## reach max depth
			root["L"] = self.__get_vote(left)
			root["R"] = self.__get_vote(right)
			return

		if len(left) <= self.min_size:  ## no more split if patition size is less than min size
			root["L"] = self.__get_vote(left) 
		else:
			root["L"] = self.__find_best_split(left)  ## build left tree
			self.__build_tree(root["L"], depth = depth + 1)
		
		if len(right) <= self.min_size:  ## no more split if patition size is less than min size
			root["R"] = self.__get_vote(right)
		else:
			root["R"] = self.__find_best_split(right)  ## build right tree
			self.__build_tree(root["R"], depth = depth + 1)

	def __find_best_split(self, dataset):
		''' find best split for dataset '''
		num_columns = len(dataset[0]) - 1
		best_gini = 999
		best_split = {"attribute": None, "threshold": None, "patitions": None}
		for att in range(num_columns): ## try each attributes
			for value in [row[att] for row in dataset]:  ## try threshold for each row of value in attributes
				left, right = self.__split(dataset, att, value)
				data_patitions = [ [x[-1] for x in left], [x[-1] for x in right] ]
				gini = self.__get_gini_score(data_patitions)
				if gini < best_gini:
					best_gini = gini
					best_split["attribute"] = att
					best_split["threshold"] = value
					best_split["patitions"] = [left, right]
		# print(best_split["attribute"])
		# print(best_split["threshold"])
		return best_split

	def __split(self, dataset, attribute, threshold):
		''' 
		split into left part and right part according to attribute and threshold 
			i.e. if data[row_ind, attribute] < threshold 
					then data[row_ind, ] belong left
				 else data[row_ind, ] belong right 
		'''
		left = []
		right = []
		for row in dataset:
			if row[attribute] < threshold:
				left.append(row)
			else:
				right.append(row)
		return left, right

	def __get_vote(self, dataset):
		''' no more split then vote for class with max number in dataset '''
		votes = [row[-1] for row in dataset]
		return max(set(votes), key = votes.count)

	def __get_gini_score(self, data_patitions):
		''' to measure entropy of a data patition used in CART
			formula: sum(j)( sum(i)(1 - p(i, j) * p(i, j) ) * size(j) / size_total )
				- p(i, j) is the proportion of class i in patition j
				- size(j) is the data size of patition j
				- size_total is the total data size
		'''
		assert type(data_patitions) is list
		assert len(data_patitions) > 0
		
		size_total = 0
		score = 0.0
		for patition in data_patitions:
			size_p = len(patition)  ## size of partition
			if size_p == 0:
				continue
			size_total += size_p
			count = {}
			for c in patition:
				if c not in count:
					count[c] = 1
				else:
					count[c] += 1
			p = [ float(count[k])/size_p for k in count.keys() ]  ## count to proportion
			score += ( 1 - sum([x*x for x in p]) ) * size_p
		if size_total == 0:
			return None
		score = score / size_total
		return score

if __name__ == '__main__':

	# dataset = [[0,0,0,1,1,1], [1,1,1,2,1,0]]
	# dataset = [[], [1,1,1,2,1,0]]

	# DT = DecisionTree()
	# print(DT.get_gini_score(dataset))

	import time
	data = pd.read_csv("data/data.csv")
	# print(data.shape)
	data = np.array(data)
	t0 = time.time()
	
	DT = DecisionTree()

	DT.fit(data)
	pprint.pprint(DT.tree)
	
	DT.predict(data)

	print("total time: ", time.time() - t0)