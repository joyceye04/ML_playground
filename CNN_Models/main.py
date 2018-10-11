'''
Author: Jing Y.
Date: Aug, 2018
'''

from CnnModel import CnnModel
# from data import Data
import configs

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
import argparse

from tensorflow import keras

img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
num_classes = 10

def prepare_mnist_data():
	data = np.load('mnist.npz')
	x_train = data['x_train']
	y_train = data['y_train']
	x_test = data['x_test']
	y_test = data['y_test']

	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	x_train /= 255
	x_test /= 255

	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	
	return x_train, y_train, x_test, y_test


def run_train_mode(data_configs, model_configs, save_path = None):

	print("init")
	# data = Data(data_configs)
	model = CnnModel(model_configs)
	
	x_train, y_train, x_test, y_test = prepare_mnist_data()
	if model.set_configs() == 'SUCCESS':
		model.build_graph(input_shape = input_shape)

		print("training model")
		model.train(x_train, y_train)

		if save_path:
			print("saving model")
			model.save_model("{}.h5".format(save_path))
	else:
		print("error: config error")

if __name__ == '__main__':
	run_train_mode(data_configs = None, model_configs = configs.model_configs)