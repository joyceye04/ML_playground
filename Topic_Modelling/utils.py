'''
Author: Jing Y.
Date: Aug, 2018
'''

import os
import string
import zipfile
import pandas as pd
import pickle

import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, brown

from sklearn.datasets import fetch_20newsgroups


PROD = os.getenv('PRODUCTION', True)
if PROD:
	nltk.download('wordnet')
	nltk.download("stopwords")
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')

STOPWORDS = set(stopwords.words('english'))
PUNCTUATIONS = set(string.punctuation) 

customized_remove_words = ['say', 'says', 'said', 'get', 'got', 'like', 'im', 'ive']

class LemmaTokenizer(object):
	def __init__(self):
		self.__lemma = WordNetLemmatizer()

	def __call__(self, articles):
		try:
			tokens_tagged = pos_tag(articles.split())
			return [self.__lemma.lemmatize(w, self.__conv(t)) for w, t in tokens_tagged]
		except Exception as err:
			print(err)
			return ['']
		
	def __conv(self, t):
		# return 'n'
		if t in ["VB", "VBG", "VBN", "VBP", "VBZ", "CD"]:
			return 'v'
		elif t in ["JJ", "JJS", "JJR"]:
			return 'a'
		else:
			return 'n'

def filter_word(word, filtered_dict = customized_remove_words, filter_by_length = 1, filter_digit = True):
	if filter_by_length > 0 and len(word) <= filter_by_length:
		return False
	if word in filtered_dict:
		return False
	if filter_digit and word.isdigit():
		return False
	
	return True

def clean_doc(doc):
	stop_free = " ".join([i for i in doc.lower().split() if i not in STOPWORDS])
	punc_free = ''.join([ch for ch in stop_free if ch not in PUNCTUATIONS])
	filtered = list(filter(lambda word: filter_word(word), punc_free.split()))
	result = " ".join(filtered)
	return result

def download_brown_data():
	print("downloading brown data from nltk.corpus")
	data = []
	for fileid in brown.fileids():
		document = ' '.join(brown.words(fileid))
		data.append(document)
	return data

def download_news_data():
	print("downloading news data from sklearn.datasets")
	dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
	documents = dataset.data
	return documents

def read_wiki_file(filepath):
	re = []
	zfile = zipfile.ZipFile(filepath)
	metadata = {}
	urlHead = "https://www.wikipedia.org/wiki/"
	ind = 0
	for finfo in zfile.infolist():
		ifile = zfile.open(finfo, 'r')
		doc = ifile.read().decode("utf-8")
		re.append(doc)
		fname = finfo.filename
		metadata[ind] = { "fname": fname, "url": urlHead + fname.split('.')[0] }
		ind += 1
	fil = filepath.split("/")[-1]
	with open("data/metadata-{}.pkl".format(fil), 'wb') as f:
		pickle.dump(metadata, f)

	return re

def read_zip_file(filepath):
	re = []
	zfile = zipfile.ZipFile(filepath)

	for finfo in zfile.infolist():
		ifile = zfile.open(finfo, 'r')
		doc = ifile.read().decode("utf-8")
		re.append(doc)

	return re

def get_metadata(data_source):
	
	fil = data_source.split("-")[-1]
	filepath = "data/metadata-{}.zip.pkl".format(fil)
	if os.path.exists(filepath):
		with open(filepath, 'rb') as f:
			metadata = pickle.load(f)
		return metadata
	else:
		print("warning: metadata is not found {}".format(filepath))
		return None

def get_data(data_source):
	if data_source == 'news':
		data = download_news_data()
	elif data_source == 'brown':
		data = download_brown_data()
	elif 'zip' in data_source:
		fil = data_source.split("-")[-1]
		if 'wiki' in fil:
			data = read_wiki_file('../shared-data/{0}.zip'.format(fil))
		else:
			data = read_zip_file('data/{0}.zip'.format(fil))
	elif 'csv' in data_source:
		fil = data_source.split("-")[-1]
		data = list(pd.read_csv("data/{0}.csv".format(fil))['text'])
	else:
		print("error: data source not recognized, support 'news', 'brown', 'zip-filename', 'csv-filename' only")
		return None
	return data

def generate_model_save_path(configs):
	data_name = configs['data_name']
	model_name = configs['type_model']
	vectorizer_name = configs['type_vectorizer']
	topic_num = configs['optional']['num_topics']
	feature_num = configs['optional']['max_features']
	path_prefix = 'save/{0}_{1}_{2}_{3}_{4}'.format(data_name, model_name, vectorizer_name, topic_num, feature_num)
	return path_prefix
