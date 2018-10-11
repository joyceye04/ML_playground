'''
Author: Jing Y.
Date: Aug, 2018
'''

import TopicModel
import TopicModelOptimizer as TMO
from flags import configs, app_configs, para_space
import utils
import log_handler
logger = log_handler.get_logger('root')

# import pprint

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type = str, help="to specify run mode")
parser.add_argument("--save", action = "store_true", default = False)
args = parser.parse_args()

def main(mode = "train"):
	
	logger.info("downloading data...")
	data = utils.get_data(configs['data_name'])
	if data is None:
		return
	logger.info("total data number: {}".format(len(data)))
	# return
	TM = TopicModel.TopicModel(configs = configs)
	
	path_prefix = utils.generate_model_save_path(configs)

	if mode == 'train':
		logger.info("training model...")
		TM.train_model(data, clean = True)

		if args.save:
			logger.info("saving model and vectorizer")
			TM.save_model(path_prefix)

		# logger.info("displaying topics...")
		# re = TM.display_topics(data)
		# print(re)

	elif mode == 'apply':
		logger.info("restoring model")
		TM.restore_model(path_prefix)
		metadata = utils.get_metadata(configs["data_name"])
		logger.info("applying model")
		docs = [""]
		response = TM.apply_model(docs, data, metadata, app_configs)
		print(response)
	
	elif mode == 'opt':
		opt_run()
	
	else:
		logger.error("mode is not recognized, support 'train', 'apply', 'opt' only")

def opt_run():

	data = utils.get_data(configs['data_name'])
	if data is None:
		return

	optimizer = TMO.TopicModelOpmitizer(configs, para_space)
	optimizer.run(data)


if __name__ == '__main__':
	main(mode = args.mode)
	