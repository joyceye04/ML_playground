'''
config file
'''
###########################
DATA_NAME = 'zip-wikiLarge'

TYPE_VEC = 'tfidf' ## opitons in ['tfidf', 'count']
TYPE_MODEL = 'nmf' ## options in ['nmf', 'lsi', 'lda']

MIN_DF = 0.0  ## min ratio for word frequency
MAX_DF = 0.5  ## max ratio for word frequency
NGRAM_MIN = 1  ## min for word gram
NGRAM_MAX = 2  ## max for word gram
MAX_FEATURES = 5000  ## max number of feature in word vectors

NUM_TOPICS = 30  ## how many topics

TOPIC_WORDS_NUM = 10 ## how many words to present each topic
TOP_WORDS_NUM = 5 ## how many keywords to present each doc
TOP_TOPIC_NUM = 5 ## how many topic to present each doc
TOP_DOC_NUM = 5 ## how many similar docs to find for each doc
DOC_PREVIEW_LEN = 200 ## how long for doc preview

##############################
## model configs
configs = {	"data_name": DATA_NAME,
			"type_vectorizer": TYPE_VEC,
			"type_model": TYPE_MODEL,
			"optional": {"max_df": MAX_DF,
						 "min_df": MIN_DF,
						 "ngram_min": NGRAM_MIN,
						 "ngram_max": NGRAM_MAX,
						 "max_features": MAX_FEATURES,
						 "num_topics": NUM_TOPICS
						}
			}

## app configs
app_configs = { "topic_words_num": TOPIC_WORDS_NUM,
				"top_words_num": TOP_WORDS_NUM,
				"top_topic_num": TOP_TOPIC_NUM,
				"top_doc_num": TOP_DOC_NUM,
				"doc_preview_len": DOC_PREVIEW_LEN
				}

## optimization configs
para_space = { "space_vec": ['tfidf'],
				"space_model": ['nmf', 'lsi', 'lda'],
				"space_num_topics": range(10,11),
				"space_max_features": [3000]				
				}
##############################