'''
Author: Jing Y.
Date: Aug, 2018
'''

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib

def generate_data_batches(series, time_step, look_forward = 1, typ = 'm2m'):
	'''
	a function to generate batch of data for RNN/LSTM/GRU models based on time series dataset
	args:
		series: a time series data with 2-dim shape (n_inputs, input_features)
		time_step: num of time step for recurrent model
		look_forward: indicate how many steps that y shall take after X(t), i.e. y(t+1), y(t+2)...y(t+look_forward)
						if type is "many to one", then look_forward = 1
		typ: choice of "m2m"(many to many) or "m2o"(many to one).
	returns:
		out_x: x_data for train or test, shape as (n_inputs, time_step, input_features)
		out_y: y_data for train or test, shape as (n_inputs, time_step, input_features) if m2m and (n_inputs, input_features) if m2o
	'''
	
	if len(series.shape) != 2:
		print("please reshape your input data as (n_inputs, input_features), \
				i.e. if n_x_features = 1, then shape should be (-1,1) ")
		return None
	
	if look_forward > time_step:
		print("")
		return None
	
	out_x = []
	out_y = []
	N = series.shape[0]
	input_features = series.shape[1]
	
	if typ == 'm2m':
		for i in range(N - time_step - look_forward + 1):
			out_x.append(series[i : i + time_step, :])
			out_y.append(series[i + look_forward: i + look_forward + time_step, :])  ## many to many
		return np.array(out_x).reshape(-1, time_step, input_features), np.array(out_y).reshape(-1, time_step, input_features)
	
	elif typ == 'm2o':
#         look_forward = 1
		for i in range(N - time_step):
			# print(i)
			out_x.append(series[i : i + time_step])
			out_y.append(series[i + time_step])  ## many to one
		return np.array(out_x).reshape(-1, time_step, input_features), np.array(out_y).reshape(-1, input_features)


class Data():
	def __init__(self, configs):
		'''
		Data init configs:
			time_step: number of time steps for rnn to recur
			look_forward: number of forward steps for y to consider after x, must be >=1
			data_type: (m2m)many to many or (m2o)many to one
			normalization: wheather data need to be normalized (use MinMaxScaler)
		'''
		self.test_ratio = configs['test_ratio']
		self.time_step = configs['time_step']
		self.look_forward = configs['look_forward']
		self.data_typ = configs['data_type']
		self.normalization = configs['normalization']
		# self.scaler = MinMaxScaler()
		self.scaler = StandardScaler(with_mean = True, with_std = True)

	def get_train_test_xy(self, data, split_test_ratio = 0.4):
		'''
		Split data into train/test according to given configs
		
		args:
			Data: 2-dim numpy array
			split_test_ratio: test data ratio
		returns:
			x_train, y_train, x_test, y_test
		'''
		data_len = data.shape[0]
		train_num = int((1-split_test_ratio) * data_len)
		train_set = data[:train_num, :]
		test_set = data[train_num: , :]

		if self.normalization:
			train_norm = self.scaler.fit_transform(train_set)
			x_train, y_train = generate_data_batches(train_norm, time_step = self.time_step, typ = self.data_typ, look_forward = self.look_forward)
			test_norm = self.scaler.transform(test_set)
			x_test, y_test = generate_data_batches(test_norm, time_step = self.time_step, typ = self.data_typ, look_forward = self.look_forward)
		else:
			x_train, y_train = generate_data_batches(train_set, time_step = self.time_step, typ = self.data_typ, look_forward = self.look_forward)
			x_test, y_test = generate_data_batches(test_set, time_step = self.time_step, typ = self.data_typ, look_forward = self.look_forward)
		print("info: training shape - X: {0}, Y: {1}".format(x_train.shape, y_train.shape))
		return x_train, y_train, x_test, y_test

	def get_apply_xy(self, apply_df):
		if self.normalization:
			apply_norm = self.scaler.transform(apply_df)
			x_apply, y_apply = generate_data_batches(apply_norm, time_step = self.time_step, typ = self.data_typ, look_forward = self.look_forward)
		else:
			x_apply, y_apply = generate_data_batches(apply_df, time_step = self.time_step, typ = self.data_typ, look_forward = self.look_forward)

		return x_apply, y_apply
	
	def get_seed_x(self, seed_df):
		if self.normalization:
			seed_norm = self.scaler.transform(seed_df)
			x_seed, _ = generate_data_batches(seed_norm, time_step = self.time_step, typ = self.data_typ, look_forward = self.look_forward)
		else:
			x_seed, _ = generate_data_batches(seed_df, time_step = self.time_step, typ = self.data_typ, look_forward = self.look_forward)

		return x_seed

	def inverse_data(self, data):
		if not self.normalization:
			return data
		return self.scaler.inverse_transform(data)

	def save_scaler(self, path):
		joblib.dump(self.scaler, path)
		print("info: scaler saved to {}".format(path))

	def load_scaler(self, path):
		self.scaler = joblib.load(path)
		print("info: scaler loaded from {}".format(path))
