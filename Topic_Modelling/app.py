'''
Author: Jing Y.
Date: Aug, 2018
'''

from flask import Flask, request, jsonify
import json

import TopicModel
from flags import configs, app_configs
import utils

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
	data = json.loads(request.get_data())
	text = data['text']
	# data = json.loads(request.get_data().decode('utf-8'))
	# text = [data['nlp']['source']]
	# print(text)
	if type(text) is not list:
		return jsonify(status = 200, erorr = "type error: please input list of text")

	re = model.apply_model(text, data = train_data, metadata = metadata, app_configs = app_configs)
	return jsonify(
			status = 200,
			response = re,
			# replies = [{'type': 'text', 'content': re}],
			info = "Please call GET /topics to check each topic key words for given topic index"
			)

@app.route('/topics', methods=['GET'])
def get_topics():
	re = model.display_topics(data = train_data, app_configs = app_configs)
	return jsonify(
			status = 200,
			response = [{'topics': re}]
			# replies = [{'type': 'text', 'content': 'response'}]
			)

def prepare():
	print("======= app warming up")
	TM = TopicModel.TopicModel(configs = None)
	print("======= restoring model")
	path_prefix = utils.generate_model_save_path(configs)
	TM.restore_model(path_prefix, app_configs)
	print("======= preparing data")
	train_data = utils.get_data(configs["data_name"])
	metadata = utils.get_metadata(configs["data_name"])
	return TM, train_data, metadata


if __name__ == '__main__':
	
	model, train_data, metadata = prepare()

	app.run(port = 5000)