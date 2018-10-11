'''
Author: Jing Y.
Date: Aug, 2018
'''

import numpy as np 
import h5py

import tensorflow as tf
from tensorflow import keras


class RnnModel():
	def __init__(self, configs):
		self._configs = configs

	def set_configs(self, configs = None):
		if not configs:
			configs = self._configs
		try:
			self.batch_size = configs['batch_size']
			self.num_epochs = configs['num_epochs']
			self.rnn_layers = configs['rnn_layers_map']
			self.dense_layers = configs['dense_layers_map']
			self.metrics = configs['metrics']
			self.set_optimizer( opt_configs = configs['optimizer'] )
			self.set_loss( loss_configs = configs['loss'] )
			return 'SUCCESS'
		except Exception as err:
			print("error: config parameters error")
			print(err)
			return 'FAIL'
			
	def set_optimizer(self, opt_configs):
		'''
		optimizer type: adam, sgd, Adagrad, Adadelta
		'''
		if opt_configs['default']:
			self.optimizer = opt_configs['type']

		else:
			typ = opt_configs['type'].lower()
			lr = opt_configs['lr']  ## lr range
			decay = opt_configs['decay']

			if typ == 'adam':
				self.optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)
			elif typ == 'sgd':
				self.optimizer = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=decay, nesterov=False)
			elif typ == 'adagrad':
				self.optimizer = keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=decay)
			elif typ == 'adadelta':
				keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=decay)
			else:
				print("error: optimizer type {} is not supported".format(typ))

	def set_early_stop(self):
		self.early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.003, verbose=1, mode='auto')

	def set_loss(self, loss_configs):
		'''
		source of loss: https://github.com/keras-team/keras/blob/master/keras/losses.py
		'''
		typ = loss_configs['type']
		loss_collections = set(["mean_squared_error", 
								"categorical_crossentropy", 
								"binary_crossentropy",
								"categorical_hinge",
								"hinge"])

		if typ in loss_collections:
			self.loss = typ
		else:
			print("error: please set loss type in one of {}".format(loss_collections))

	def build_graph(self, input_features = 1):
		self.model = keras.models.Sequential()
		prev_unit = input_features
		for cell in self.rnn_layers:
			typ = cell['type']
			unit = cell['unit']
			seq = cell['return_seq']  ## return_sequence
			act = cell['activation']
			if typ == 'LSTM':
				self.model.add(keras.layers.LSTM(units = unit, 
									activation = act, 
									return_sequences = seq, ## many to many if True; many to one if False
									stateful = False, ##  if False it will generate a new state, if True then the state from last training will be used as the initial state
									input_shape=(None, prev_unit)))
			elif typ == 'GRU':
				self.model.add(keras.layers.GRU(units = unit, 
									activation = act, 
									return_sequences = seq, ## many to many if True; many to one if False
									stateful = False, ##  if False it will generate a new state, if True then the state from last training will be used as the initial state
									input_shape=(None, prev_unit)))
			prev_unit = unit
		
		for cell in self.dense_layers:
			unit = cell['unit']
			act = cell['activation']
			self.model.add(keras.layers.Dense(units = unit, 
											  activation = act,
											  use_bias = True,
										))
		
		self.model.compile( optimizer = self.optimizer, 
							loss = self.loss,
							metrics = self.metrics
					 )
		print("info: model is successfully built and architecture shown as below")
		self.print_summary()

	def print_summary(self):
		if self.model:
			self.model.summary()
		else:
			print("error: model is not set up")

	def train(self, x_train, y_train):
		m = self.model.fit(x_train, y_train, 
					  batch_size = self.batch_size, 
					  epochs = self.num_epochs,
			#           validation_split=0.15,
			#           callbacks=[self.early_stop]
					 )
		self.history = m.history

	def predict(self, x_test):
		return self.model.predict(x_test)

	def auto_predict(self, x_seed, timestep, iter_ = 10):
		x_in = x_seed.copy()
		for i in range(iter_):
			y_out = np.array([self.model.predict(x_in[:, -timestep:, :])])
			x_in = np.concatenate([x_in,y_out], axis = 1)

		return x_in

	def save_model(self, path):
		if path.split(".")[-1] == 'h5':
			self.model.save(path)
			print("info: model saved to {}".format(path))
		else:
			print("error: current saving support h5 format only") 

	def load_model(self, path):
		self.model = None

		if path.split(".")[-1] == 'h5':
			self.model = keras.models.load_model(path)
			print("info: model loaded from {}".format(path))
		else:
			print("error: current saving support h5 format only")

	def get_loss_history(self):
		return self.history['loss']
