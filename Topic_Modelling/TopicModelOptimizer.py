'''
Author: Jing Y.
Date: Aug, 2018
'''

import TopicModel

import flags as FLAGS
import utils

import itertools

class TopicModelOpmitizer(object):
	def __init__(self, configs, para_space):
		
		self.model = TopicModel.TopicModel(configs = None)
		self.configs = configs

		self.type_vectorizer_space = para_space['space_vec']
		self.type_model_space = para_space['space_model']
		self.num_topics_space = para_space['space_num_topics']
		self.max_features_space = para_space['space_max_features']

	def iter_configs(self):
		space = [self.type_vectorizer_space, self.type_model_space, self.num_topics_space, self.max_features_space]
		config_map = list(itertools.product(*space))
		return config_map

	def run(self, data):
		config_map = self.iter_configs()
		for tpe_v, tpe_m, n_topics, max_fea in config_map:
			self.configs['type_vectorizer'] = tpe_v
			self.configs['type_model'] = tpe_m
			self.configs['optional']['num_topics'] = n_topics
			self.configs['optional']['max_features'] = max_fea
			self.model.set_up(self.configs)
			print(self.configs)
			self.model.train_model(data)


if __name__ == '__main__':
	
	configs = FLAGS.configs
	space = FLAGS.para_space

	print("======= downloading data")
	data = utils.get_data(configs['data_name'])
	
	TMO = TopicModelOpmitizer(configs, space)
	TMO.run(data)

	