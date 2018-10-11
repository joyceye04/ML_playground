'''
Author: Jing Y.
Date: Aug, 2018
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

import pandas as pd
import numpy as np
import nltk
import io
import random
import collections
from langdetect import detect

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# nltk.download('stopwords')
stop_words_en = set(stopwords.words('english'))
# stop_words_de = set(stopwords.words('german'))

FILTERED_DICT = stop_words_en


def load_vectors(filepath, header = False):
	'''
	load a vector file with format as:
		word_1 321 412 1 3 ...
		word_2 12 23 432 32 ...
		...
	if the first line gives the overall number of words, then header is True;

	return a dictionary with key value pair as (word, vector)
	'''
	fin = io.open(filepath, 'r', encoding='utf-8', newline='\n', errors='ignore')
	if header:
		n, d = map(int, fin.readline().split())
	data = {}
	for line in fin:
		tokens = line.rstrip().split(' ')
		data[tokens[0]] = list(map(float, tokens[1:]))
	return data


def filter_word(word, filtered_dict = FILTERED_DICT, filter_by_length = 0, filter_digit = True):
	'''
	filter out word if
		its length is less than filter_by_length
		it is included in filtered_dict like stopwords
		set filter_digit true and it is digit number
	'''
	if filter_by_length > 0 and len(word) <= filter_by_length:
		return False
	
	if word in filtered_dict:
			return False
	
	if filter_digit and word.isdigit():
		return False
	
	return True


def text_to_tokens(text, _filter = True):
	'''
	'''
	reg = r'\w+'
	tokenizer = RegexpTokenizer(reg)
	tokens = tokenizer.tokenize(text)
	if _filter:
		tokens = list(filter(lambda word: filter_word(word), tokens))
	return tokens


def tokens_to_vectors(tokens, look_up, dim_vectors, aggregation = 'mean'):
	'''
	'''
	if not look_up:
		print("error: please define look up dictionary")
		return None
	vec = []
	c = 0
	for t in tokens:
		if t in look_up.keys():
			c += 1
			vec.append(list(look_up[t]))
	try:
		vec = np.array(vec).reshape(c, dim_vectors)
	except:
		print("error: please check dimension of vectors")
		return vec

	if aggregation == 'raw':
		return vec
	elif aggregation == 'mean':
		return np.mean(vec, axis = 0)
	elif aggregation == 'max':
		return np.max(vec, axis = 0)
	else:
		print("error: aggregation is not defined correctly. It should be - raw, mean, max")
		return None


def text_to_vectors(text, look_up, dim_vectors):
	'''
	'''
	if type(text) is list:
		num = len(text)
		vec_out = []
		for text_each in text:
			tokens = text_to_tokens(text_each)
			vec = tokens_to_vectors(tokens, look_up, dim_vectors)
			vec_out.append(vec)
		return np.array(vec_out).reshape(num, dim_vectors)
	else:
		print("error: please give text input as list")
		return None


def one_hot_encoder(labels):
	'''
	'''
	label_list = list(set(list(labels)))
	num = len(label_list)
	matrix = np.eye(num)
	encoder = {}
	decoder = {}
	for i in range(num):
		encoder[label_list[i]] = matrix[i]
		decoder[i] = label_list[i]
	return encoder, decoder