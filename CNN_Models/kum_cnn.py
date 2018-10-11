'''
Author: Jing Y.
Date: Aug, 2018

the code below is to simply implement paper of
Convolutional Neural Networks for Sentence Classification, Yoon Kim, 2014
https://arxiv.org/pdf/1408.5882.pdf

the pre-trained word vector glove can be downloaded from: https://nlp.stanford.edu/projects/glove/

the dataset used can be downloaded from: 
Amazon/Yelp/imdb: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
TREC: http://cogcomp.org/Data/QA/QC/
'''

import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import utils

SENTENCE_LENGTH = 5 ## take average sentence length as row of sentense representation
WORD_VEC_DIM = 300 ## take word vector dim as column of sentense representation
OUT_DIM = 6 ## output dim of y

N_GRAMS_LIST = [2, 3] ## a list of n_grams to derive multiple convolutional layers
NUM_FILTERS = 32  ## number of filters for each convolutional layer
NUM_DENSE_LAYERS = 2 ## number of dense layers after convolutional layers concatenation
DROP_RATE = 0.2 ## drop rate after dense layers

LR = 0.1

DATA_TYPE = "MULTI"
# DATA_TYPE = "BINARY"
# LOSS_TYPE = "binary_crossentropy"
LOSS_TYPE = 'categorical_crossentropy'

NUM_EPOCHS = 100
BATCH_SIZE = 50
TEST_SIZE  = 0.2

def one_hot_encoder(labels):
    label_list = list(set(list(labels)))
    num = len(label_list)
    matrix = np.eye(num)
    encoder = {}
    decoder = {}
    for i in range(num):
        encoder[label_list[i]] = matrix[i]
        decoder[i] = label_list[i]
    return encoder, decoder

def prune_text(input_text, output_len = SENTENCE_LENGTH, dim_vectors = WORD_VEC_DIM):
	'''
	to prepare vector array for cnn model
	'''
	tokens = utils.text_to_tokens(input_text, False)
	vec = utils.tokens_to_vectors(tokens, glove_vec, dim_vectors = dim_vectors, aggregation = 'raw')
	diff = output_len - vec.shape[0]
	if diff > 0:
		out_vec = np.concatenate([vec, np.zeros((diff, dim_vectors))], axis = 0)
	else:
		out_vec = vec[:output_len]
	return out_vec, len(tokens), vec.shape[0]


def build_kum_cnn_graph(sent_len, word_vec, out_dim, filters = 64, n_grams = [2,3], num_dense_layers = 3):
	'''
	args:
		sent_len: length of input sentense. if raw sentense is less than it, using zero padding, else cut down to it.
		word_vec: dim of word vector embedding using pre-trained glove or word2vec model
		out_dim: dim of output y
		filters: filters for Convolutional layers
		n_grams: list of ngram for Convolutional layers kernal. each will generate one cell output. details can be referred from paper
		num_dense_layers: to decide how many dense layers after concatenating all Convolutional layers output
	returns:
		Keras Model
	'''
	inputs = keras.layers.Input(shape=(sent_len, word_vec, 1))
	merged_layer = []
	for h in n_grams:
		conv_layer = keras.layers.Conv2D(filters, (h, word_vec), activation='relu')(inputs)
		pool_layer = keras.layers.MaxPooling2D(pool_size=(sent_len-h+1, 1))(conv_layer)
		merged_layer.append(pool_layer)
	concat_layer = keras.layers.concatenate(merged_layer)
	flatten_layer = keras.layers.Flatten()(concat_layer)
	in_ = flatten_layer
	prev_units = filters * len(n_grams)
	for _ in range(num_dense_layers - 1):
		prev_units /= 2
		dense_layer = keras.layers.Dense(prev_units, 
										activation='relu', 
										kernel_regularizer = keras.regularizers.l2(0.01),
										# activity_regularizer = keras.regularizers.l1(0)
										)(in_)
		drop_layer = keras.layers.Dropout(DROP_RATE)(dense_layer)
		in_ = drop_layer
	
	outputs = keras.layers.Dense(out_dim, activation = 'softmax')(in_)
	
	model = keras.models.Model(inputs = inputs, outputs = outputs)
	model.summary()
	return model


########### prepare data ############
print("================ loading vectors")
glove_vec = utils.load_vectors('data/glove.6B/glove.6B.{0}d.txt'.format(WORD_VEC_DIM))

print("================ preparing data")
# data = pd.read_csv("data/amazon_cells_labelled.txt" ,sep = '\t', header = None) ## 6 words, 1000 records
# data = pd.read_csv("data/yelp_labelled.txt" ,sep = '\t', header = None) ## 6 words, 1000 records
# data = pd.read_csv("data/imdb_labelled.txt", sep = '\t', header = None) ## 11 words, 748 records
data = pd.read_csv("data/QA.csv")
# data.columns = ['text', 'Category']
data = data.dropna()

x_array = np.array([prune_text(s)[0] for s in data.text])

lens = np.array([prune_text(s)[1:] for s in data.text])
print(np.mean(lens, axis = 0))

x_array = x_array.reshape(-1, SENTENCE_LENGTH, WORD_VEC_DIM, 1)

if DATA_TYPE == 'MULTI':
	encoder, decoder = one_hot_encoder(data.Category)
	y_array = np.array([encoder[l] for l in data.Category])
if DATA_TYPE == 'BINARY':
	y_array = np.array(data.Category).reshape(-1, OUT_DIM)

x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size = TEST_SIZE, random_state = 0)
#######################################################

############ train #############
print("================ buidling model")
m = build_kum_cnn_graph(sent_len = SENTENCE_LENGTH, 
						word_vec = WORD_VEC_DIM, 
						out_dim = OUT_DIM, 
						filters = NUM_FILTERS, 
						n_grams = N_GRAMS_LIST,
						num_dense_layers = NUM_DENSE_LAYERS)

m.compile( optimizer = keras.optimizers.Adadelta(lr = LR, rho = 0.95, epsilon = None, decay = 0.0), 
			loss = LOSS_TYPE,
			metrics =  ['accuracy']
		)

early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0, patience = 30, verbose=1, mode='auto')

m.fit( x_train, y_train, 
		epochs = NUM_EPOCHS, 
		batch_size = BATCH_SIZE, 
		validation_split = 0.3,
		callbacks = [early_stop]
	)
######################################

################ test ################
print("================ testing")
y_pred_raw = m.predict(x_test)
if DATA_TYPE == 'MULTI':
	y_pred = [decoder[np.argmax(l)] for l in y_pred_raw]
	y_true = [decoder[np.argmax(l)] for l in y_test]
if DATA_TYPE == 'BINARY':
	y_pred = [1 if x>= 0.5 else 0 for x in y_pred_raw]
	y_true = y_test
print(np.sum([y_true[i] == y_pred[i] for i in range(len(y_true))] ) / len(y_true))
print(classification_report(y_true, y_pred))
######################################