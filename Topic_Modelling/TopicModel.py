'''
Author: Jing Y.
Date: Aug, 2018
'''

import time
import numpy as np
import pickle
import log_handler
logger = log_handler.get_logger(__name__)

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.externals import joblib

import flags as FLAGS
import utils


class TopicModel():

	def __init__(self, configs = None):
		'''
		model initialize
		'''
		self.__configs = configs
		if self.__configs:
			logger.info("initializing with input configs")
			self.set_up(self.__configs) ## init with initial configs
		else:
			logger.info("info: no initial configs found. Options: \n1) Call set_up(your_configs) to initialize \n2) Call restore_model(your_path) to reuse saved ones")

	def __clear_memory(self):
		''' 
		clear vectorizer and model state
		'''
		self.__vectorizer = None
		self.__model = None
		self.__doc_matrix = None

	def set_up(self, configs): 
		'''
		configs can be from initial configs or new configs after initialize class
		'''
		self.__clear_memory()  ## clear model status before set up new configs

		if not configs or type(configs) is not dict:
			logger.error("cannot initialize without configs and configs must be type of dictionary")

		## ------------- optional configs --------------
		if 'optional' in configs.keys():
			logger.info("setting optional configs")
			ops = configs['optional']
			if 'max_df' in ops.keys():
				self.__max_df = ops['max_df']
			else:
				self.__max_df = 0.95  ## default
			if 'min_df' in ops.keys():
				self.__min_df = ops['min_df']
			else:
				self.__min_df = 0.0
			if 'ngram_min' in ops.keys():
				self.__ngram_min = ops['ngram_min']
			else:
				self.__ngram_min = 1
			if 'ngram_max' in ops.keys():
				self.__ngram_max = ops['ngram_max']
			else:
				self.__ngram_max = 1
			if 'num_topics' in ops.keys():
				self.__num_topics = ops['num_topics']
			else:
				self.__num_topics = 10
			if 'max_features' in ops.keys():
				self.__max_features = ops['max_features']
			else:
				self.__max_features = None
		## ------------- END of optional configs --------------
		
		if "type_vectorizer" in configs.keys():
			self.set_vectorizer(configs['type_vectorizer'])
			logger.info("vectorizer type has been set as: {}".format(configs['type_vectorizer']))
		else:
			logger.error("cannot init vectorizer without correct config key - type_vectorizer")

		if "type_model" in configs.keys():
			self.set_model(configs['type_model'])
			logger.info("model type has been set as: {}".format(configs['type_model']))
		else:
			logger.error("cannot init model without correct config key - type_model")

	def set_vectorizer(self, type_vectorizer):
		'''
		set vectorizer
		'''
		if type_vectorizer == 'tfidf':
			self.__vectorizer = TfidfVectorizer(max_df = self.__max_df, 
										 min_df = self.__min_df,
										 max_features = self.__max_features, 
										 tokenizer = utils.LemmaTokenizer(),
										 ngram_range = (self.__ngram_min, self.__ngram_max),
										 stop_words = 'english',
										 lowercase = True, 
										 # token_pattern = '[a-zA-Z\-][a-zA-Z\-]{2,}'
										 )
		
		elif type_vectorizer == 'count':
			self.__vectorizer = CountVectorizer(max_df = self.__max_df, 
										 min_df = self.__min_df, 
										 max_features = self.__max_features,
										 tokenizer = utils.LemmaTokenizer(),
										 ngram_range = (self.__ngram_min, self.__ngram_max),
										 stop_words = 'english', 
										 lowercase = True, 
										 # token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}'
										 )
		# elif type_vectorizer == 'hash':
		# 	self.__vectorizer = HashingVectorizer(ngram_range = (self.__ngram_min, self.__ngram_max),
		# 										lowercase = True,
		# 										tokenizer = utils.LemmaTokenizer(),
		# 										stop_words = 'english',
		# 										n_features = self.__max_features,
		# 										non_negative = True
		# 										)
		else:
			logger.error("type of vectorizer has to be count or tfidf")
			return None

	def set_model(self, type_model):
		'''
		set model
		'''
		if type_model == 'lda':
			self.__model = LatentDirichletAllocation(n_topics = self.__num_topics, 
													doc_topic_prior=None, 
													max_iter = 10, 
													learning_method='online',
													topic_word_prior=None, 
													learning_decay=0.7, 
													learning_offset=10.0, 
													perp_tol=0.1, 
													mean_change_tol=0.001, 
													max_doc_update_iter=100)

		elif type_model == 'nmf':
			self.__model = NMF(n_components = self.__num_topics,
								init=None, ## 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar'
								solver='cd', ## 'cd' | 'mu'
								beta_loss='frobenius', ##  'frobenius' | 'kullback-leibler' | 'itakura-saito'
								tol=0.0001,
								max_iter=200,
								alpha=0.0,
								l1_ratio=0.0)
		
		elif type_model == 'lsi':
			self.__model = TruncatedSVD(n_components = self.__num_topics,
										algorithm='randomized',
										n_iter=5,
										tol=0.0)

		else:
			logger.error("type of model has to be lda, nmf or lsi")
			return None
	
	def get_vectorizer(self):
		return self.__vectorizer

	def get_model(self):
		return self.__model
	
	def get_feature_names(self):
		if self.__vectorizer:
			return self.__vectorizer.get_feature_names()
		else:
			logger.error("vectorizer state is empty")

	def get_feature_matrix(self):
		if self.__model:
			return self.__model.components_
		else:
			logger.error("model state is empty")
	
	def get_topics(self, topic_words_num):
		'''
		get topics as dictionary: {"topic_ind": "a list of topic_keywords"}
		'''
		feature_matrix = self.get_feature_matrix() ## (num_topic, num_features)
		feature_names = self.get_feature_names()
		
		topic_dict = {}
		for topic_ind, topic in enumerate(feature_matrix):
			keywords = ", ".join([feature_names[i] for i in topic.argsort()[:-topic_words_num - 1:-1]])
			topic_dict[topic_ind] = keywords
		return topic_dict

	def train_model(self, data, clean = True):
		if type(data) is not list:
			logger.error("type of data must be list")
		
		if clean:
			data = [utils.clean_doc(d) for d in data]

		self.__doc_matrix = self.__vectorizer.fit_transform(data)
		
		self.__weight_matrix = self.__model.fit_transform(self.__doc_matrix)  ## (num_train_doc, num_topics)
		# loss = self._compute_loss()

	def update_model(self, new_data):
		pass

	def apply_model(self, text_inputs, data, metadata, app_configs):
		'''
		apply model on list of text inputs
		'''
		t0 = time.time()
		if not self.__vectorizer or not self.__model:
			logger.error("vectorizer or model state empty")
			return None
		
		if type(text_inputs) is not list:
			logger.error("data inputs must be list")
			return None
		
		topic_words_num = app_configs['topic_words_num']
		top_words_num = app_configs['top_words_num']
		top_topic_num = app_configs['top_topic_num']
		top_doc_num = app_configs['top_doc_num']
		doc_preview_len = app_configs['doc_preview_len']

		if not self.__topic_dict:
			self.__topic_dict = self.get_topics(topic_words_num)

		res = []
		for text in text_inputs:
			re = self.predict_text(text, data, metadata, top_words_num, top_topic_num, top_doc_num, doc_preview_len)
			res.append({"inputs": text, "result": re, "latency(s)": time.time() - t0})
		return res

	def predict_text(self, text_input, data, metadata, top_words_num, top_topic_num, top_doc_num, doc_preview_len):
		'''
		predict each doc top keywords, top toipcs, top related docs in existing dataset (or train dataset)
		'''
		if not self.__vectorizer or not self.__model:
			logger.error("vectorizer or model state empty")
			return None
		
		if type(text_input) is not str:
			logger.error("data inputs must be string")
			return None
		
		### to find top weighted key words in one doc
		apply_doc_matrix_raw = self.__vectorizer.transform([utils.clean_doc(text_input)])
		apply_doc_matrix_raw = np.array(apply_doc_matrix_raw.todense())  ## (1, num_features)
		top_keywords_list = self._get_top_weighted_keywords(apply_doc_matrix_raw, top_words_num)
		###########################################
		### to find top weighted toipcs in one doc
		apply_weight_matrix = self.__model.transform(apply_doc_matrix_raw) ## (1, num_topic)
		top_topic_list = self._get_top_weighted_topics(apply_weight_matrix, top_topic_num)
		###########################################
		### to find top related docs by keywords and by topics
		if not data:
			logger.warning("data is empty, cannot find most similar docs")
		else:
			## find top docs by topic
			if len(top_topic_list) > 0:
				top_docs_ind_topic = self._get_most_similar_docs(apply_weight_matrix[0], self.__weight_matrix, top_doc_num)
				top_docs_topic = self._get_top_docs(top_docs_ind_topic, data, metadata, doc_preview_len)
			else: 
				top_docs_topic = []
			## find top docs by keywords
			if len(top_keywords_list) > 0:
				top_docs_ind_feature = self._get_most_similar_docs(apply_doc_matrix_raw[0], self.__doc_matrix, top_doc_num)
				top_docs_feature = self._get_top_docs(top_docs_ind_feature, data, metadata, doc_preview_len)
			else:
				top_docs_feature = []
		###########################################
		re = { 
				"keywords (score)": top_keywords_list,
				"key topics (score)": top_topic_list,
				"top similar docs by keywords (preview)": top_docs_feature,
				"top similar docs by topic (preview)": top_docs_topic
			 }
		return re

	def display_topics(self, data, app_configs):
		feature_matrix = self.get_feature_matrix() ## (num_topic, num_features)
		feature_names = self.get_feature_names()
		
		top_words_num = app_configs['top_words_num']
		top_doc_num = app_configs['top_doc_num']
		doc_preview_len = app_configs['doc_preview_len']

		re = {}
		for topic_ind, topic in enumerate(feature_matrix):
			topic_key = "topic #{}".format(topic_ind)
			keywords = ", ".join([feature_names[i] for i in topic.argsort()[:-top_words_num - 1:-1]])
			top_doc_ind = self.__weight_matrix[:,topic_ind].argsort()[:-top_doc_num-1:-1]
			top_docs = []
			# for doc_ind in top_doc_ind:
			# 	each_doc = {
			# 				"key": "doc #{}".format(doc_ind),
			# 				"text-preview": data[doc_ind][:doc_preview_len], 
			# 				"score": round(self.__weight_matrix[doc_ind, topic_ind], 2) 
			# 				}
			# 	top_docs.append(each_doc)
			re[topic_key] = {
							"key words": keywords,
							"top related docs in training dataset (preview)": top_docs
							}
		return re

	def save_model(self, fpath):
		logger.info("model saved to path {0}".format(fpath))
		self.__dump_model(fpath + '_model.pkl')
		self.__dump_vectorizer(fpath + '_vec.pkl')
		self.__dump_array(fpath + '_array')

	def restore_model(self, fpath, app_configs):
		logger.info("model restored from {0}".format(fpath))
		self.__clear_memory()
		self.__load_model(fpath + '_model.pkl')
		self.__load_vectorizer(fpath + '_vec.pkl')
		self.__load_array(fpath + '_array.npy')
		
		logger.info("restore model state matrix")
		self.__weight_matrix = self.__model.transform(self.__doc_matrix) ## (num_doc, num_topic)
		self.__topic_dict = self.get_topics(app_configs['topic_words_num'])
	
	def _get_most_similar_docs(self, x, z, top_doc_num):
		dists = euclidean_distances(x.reshape(1, -1), z)
		pairs = enumerate(dists[0])
		most_similar = sorted(pairs, key=lambda item: item[1])[:top_doc_num]
		return most_similar

	def _get_top_docs(self, top_docs_ind, data, metadata, doc_preview_len):
		'''
		get top related docs content
		'''
		top_docs = []
		for doc_ind, score in top_docs_ind:
			doc_url = ''
			if metadata:
				doc_url = metadata[doc_ind]['url']
			each_doc = {
						"text-preview": data[doc_ind][:doc_preview_len],
						"url": doc_url,
						"similarity": round(score, 2),
						"key": "doc #{}".format(doc_ind)
						}
			top_docs.append(each_doc)
		return top_docs
	
	def _get_top_weighted_keywords(self, doc_matrix, top_words_num, threshold = 0.5):
		'''
		sort keywords weight in each doc matrix (#doc, #features)
		'''
		feature_names = self.get_feature_names()
		top_word_ind = doc_matrix[0].argsort()[:-top_words_num - 1:-1]
		top_words = [feature_names[i] for i in top_word_ind]
		top_words_score = doc_matrix[0, top_word_ind]
		top_words_list = []
		for word, score in zip(top_words, top_words_score):
			if score >= threshold:
				top_words_list.append("{0} ({1})".format(word, round(score, 2)))
	
		return top_words_list
	
	def _get_top_weighted_topics(self, apply_weight_matrix, top_topic_num, threshold = 0.1):
		'''
		sort topic weight in each weight matrix (#doc, #topic)
		'''
		top_topic_ind = apply_weight_matrix[0].argsort()[:-top_topic_num - 1:-1]
		topic_score = apply_weight_matrix[0, top_topic_ind]
		top_topic_list = []
		for topic, score in zip(top_topic_ind, topic_score):
			if score >= threshold:
				top_topic_list.append("topic #{0}: {1} (score: {2})".format(topic, self.__topic_dict[topic], round(score, 2)))

		return top_topic_list
	
	def _compute_loss(self):
		feature_matrix = self.get_feature_matrix() ## (num_topic, num_features)
		doc_matrix = np.dot(self.__weight_matrix, feature_matrix)
		return np.sum(np.square(self.__doc_matrix - doc_matrix))

	def __dump_model(self, fpath):
		joblib.dump(self.__model, fpath)
	
	def __load_model(self, fpath):
		self.__model = joblib.load(fpath)
	
	def __dump_array(self, fpath):
		np.save(fpath, self.__doc_matrix)

	def __load_array(self, fpath):
		self.__doc_matrix = np.load(fpath)[()]
	
	def __dump_vectorizer(self, fpath):
		with open(fpath, 'wb') as f:
			pickle.dump(self.__vectorizer, f)
	
	def __load_vectorizer(self, fpath):
		with open(fpath, 'rb') as f:
			self.__vectorizer = pickle.load(f)
	


if __name__ == '__main__':
	
	configs = {"type_vectorizer": 'tfidf',  ## opitons = [tfidf, count]
				"type_model": 'nmf', ## options = [nmf, lda, lsi]
				"optional": {'max_df': FLAGS.MAX_DF,
							'min_df': FLAGS.MIN_DF,
							'max_features': FLAGS.MAX_FEATURES,
							'num_topics': FLAGS.NUM_TOPICS
							}
			  }
	
	TM = TopicModel(configs = configs)