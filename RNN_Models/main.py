'''
Author: Jing Y.
Date: Aug, 2018
'''

from model import RnnModel
from data import Data
import configs

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode",  type = str, help="to specify mode from train, apply, auto")
args = parser.parse_args()

def run_apply_mode(df, data_configs, model_configs, save_path):

	print("init")
	data = Data(data_configs)
	model = RnnModel(model_configs)
	
	print("loading scaler and model")
	data.load_scaler("{}.scale".format(save_path))
	model.load_model("{}.h5".format(save_path))

	x_apply, y_apply = data.get_apply_xy(apply_df = df)

	y_apply_pred = model.predict(x_apply)
	y_apply_est = data.inverse_data(y_apply_pred)
	y_apply_raw = data.inverse_data(y_apply)

	plt.clf()
	plt.title("apply raw vs estimate")
	target_ind = 0
	plt.plot(y_apply_raw[:,target_ind], label = 'y_apply_raw')
	plt.plot(y_apply_est[:,target_ind], label = 'y_appy_est')
	plt.legend()
	plt.show()

def run_auto_apply_mode(df, data_configs, model_configs, save_path, pred_size = 500):

	print("init")
	data = Data(data_configs)
	model = RnnModel(model_configs)
	
	print("loading scaler and model")
	data.load_scaler("{}.scale".format(save_path))
	model.load_model("{}.h5".format(save_path))

	seed_size = data_configs['time_step']
	st_ind = 0
	x_seed = data.get_seed_x(seed_df = df[st_ind:seed_size+st_ind+1, :])
	self_pred = model.auto_predict(x_seed, seed_size, pred_size)[0]
	self_pred_est = data.inverse_data(self_pred)

	plt.clf()
	plt.title("auto apply raw vs estimate")
	target_ind = 0
	plt.plot(df[st_ind:st_ind+pred_size+seed_size,target_ind], label = 'raw')
	plt.plot(self_pred_est[:,target_ind], label = 'auto_pred')
	plt.legend()
	plt.show()

def run_train_mode(df, data_configs, model_configs, save_path = None):

	print("init")
	data = Data(data_configs)
	model = RnnModel(model_configs)
	
	x_train, y_train, x_test, y_test = data.get_train_test_xy(df, split_test_ratio = 0.4)

	if model.set_configs() == "SUCCESS":
		model.build_graph(input_features = df.shape[1])

		print("training model")
		model.train(x_train, y_train)
		plt.clf()
		loss_history = model.get_loss_history()
		plt.title("loss history")
		plt.plot(loss_history)
		plt.show()

		if save_path:
			print("saving model")
			data.save_scaler("{}.scale".format(save_path))
			model.save_model("{}.h5".format(save_path))

		y_test_pred = model.predict(x_test)
		y_test_est = data.inverse_data(y_test_pred)
		y_test_raw = data.inverse_data(y_test)

		y_train_pred = model.predict(x_train)
		y_train_est = data.inverse_data(y_train_pred)
		y_train_raw = data.inverse_data(y_train)

		plt.clf()
		plt.title("train raw vs estimate")
		target_ind = 0
		plt.plot(y_train_est[:,target_ind], label = 'y_train_est')
		plt.plot(y_train_raw[:,target_ind], label = 'y_train_raw')
		plt.legend()
		plt.show()

		plt.clf()
		plt.title("test raw vs estimate")
		plt.plot(y_test_raw[:,target_ind], label = 'y_test_raw')
		plt.plot(y_test_est[:,target_ind], label = 'y_test_est')
		plt.legend()
		plt.show()
	else:
		print("error: config error")

def generate_path_name(pre, data_configs, model_configs):
	dt = str(data_configs['data_type'])
	ts = str(data_configs['time_step'])
	m = '_'.join([ x['type']+str(x['unit']) for x in model_configs['rnn_layers_map'] ])

	name = '.'.join([pre, dt, m, ts])
	return name

if __name__ == '__main__':

	x = np.sin(np.arange(500) * 0.05)
	df = x.reshape(-1, 1)
	pre = 'save/sin'

	save_path = generate_path_name(pre, configs.data_configs, configs.model_configs)
	print("info: saving path is ", save_path)

	if args.mode == 'train':
		run_train_mode(df, configs.data_configs, configs.model_configs, save_path)

	if args.mode == 'apply':
		run_apply_mode(df, configs.data_configs, configs.model_configs, save_path)
	
	if args.mode == 'auto':
		run_auto_apply_mode(df, configs.data_configs, configs.model_configs, save_path)