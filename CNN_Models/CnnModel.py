'''
Author: Jing Y.
Date: Aug, 2018
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py

class CnnModel():
	def __init__(self, configs):
		self._configs = configs

	def set_configs(self, configs = None):
		if not configs:
			configs = self._configs
		try:
			self.batch_size = configs['batch_size']
			self.num_epochs = configs['num_epochs']
			self.cnn_layers = configs['cnn_layers_map']
			self.metrics = configs['metrics']
			self.set_optimizer( opt_configs = configs['optimizer'] )
			self.set_loss( loss_configs = configs['loss'] )
			return 'SUCCESS'
		except Exception as err:
			print("error: config parameters error")
			print(err)
			return 'FALI'
			
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
				self.optimizer = keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=decay)
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

	def build_graph(self, input_shape):
		self.model = keras.models.Sequential()
		for layer in self.cnn_layers:
			typ = layer['type'].lower()
			if typ == 'conv2d':
				f = layer['filters']
				kernals = layer['kernal_size']
				act = layer['activation']
				if layer['input']:  
					## first layer shall give input_shape
					self.model.add(keras.layers.Conv2D( filters = f, 
														kernel_size = kernals, 
														activation = act,
														strides = (1, 1), 
														padding = 'valid',
														input_shape = input_shape
									))
				else:
					self.model.add(keras.layers.Conv2D( filters = f, 
														kernel_size = kernals, 
														activation = act,
														strides = (1, 1), 
														padding = 'valid',
									))
			elif typ == 'maxpooling1d':
				pool = layer['pool_size']  ## pool_size 1dim
				self.model.add(keras.layers.MaxPooling1D( pool_size = pool,
														strides = None, 
														padding = 'valid'  ## valid or same
								))
			elif typ == 'maxpooling2d':
				pool = layer['pool_size']  ## pool_size 2dim
				self.model.add(keras.layers.MaxPooling2D( pool_size = pool, 
														strides = None, 
														padding = 'valid'  ## valid or same
								))
			elif typ == 'maxpooling3d':
				pool = layer['pool_size']  ## pool_size 3dim
				self.model.add(keras.layers.MaxPooling3D( pool_size = pool, 
														strides = None, 
														padding = 'valid'  ## valid or same
								))
			elif typ == 'dropout':
				r = layer['rate']
				self.model.add(keras.layers.Dropout(r))
			elif typ == 'flatten':
				self.model.add(keras.layers.Flatten())
			elif typ == 'dense':
				units = layer['units']
				act = layer['activation']
				self.model.add(keras.layers.Dense( units, activation = act))
			else:
				print("error: layer type is not supported")
				return

		self.model.compile( optimizer = self.optimizer, 
							loss = self.loss,
							metrics =  self.metrics
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
			          validation_split = 0.15,
			#           callbacks=[self.early_stop],
						verbose = 1,
					 )
		self.history = m.history

	def predict(self, x_test):
		return self.model.predict(x_test)

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
		## keys: loss, acc, val_loss, val_acc per epoch
		return self.history['loss']

	def evaluate(self, x_test, y_test):
		score = self.model.evaluate(x_test, y_test, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])