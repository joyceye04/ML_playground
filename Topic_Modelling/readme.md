# Topic Modelling

## Introduction

a general topic modelling tool for both research and dev purpose

## What is Available

#### Available Datasets for Test

- nltk.corpus.brown dataset: DATA_NAME = 'brown' 


- sklearn.datasets.fetch_20newsgroups dataset: DATA_NAME = 'news'
- csv data: DATA_NAME = 'csv-\$filename\$' 
  - location: ./data/\$filename\$.csv
  - format: data in one column "text"
- zip data: DATA_NAME = 'zip-\$filename\$' 
  - location: ./data/\$filename\$.zip
  - format: contains list of .txt file as each doc

#### Available Models

- nmf
- lda
- lsi

#### Available Vectors

- tfidf
- count




## How to Train the Model

write configs in flags.py and run `python main.py --mode train`

```python
configs = {	
	"data_name": DATA_NAME,  ## options in ['news', 'brown', 'zip-<filename>', 'csv-<filename>']
	"type_vectorizer": TYPE_VEC, ## opitons in ['tfidf', 'count']
	"type_model": TYPE_MODEL, ## options in ['nmf', 'lsi', 'lda']
	"optional": {'max_df': MAX_DF,
                 'min_df': MIN_DF,
                 'ngram_min': NGRAM_MIN,
                 'ngram_max': NGRAM_MAX,
                 'max_features': MAX_FEATURES,
                 'num_topics': NUM_TOPICS
			}
	}
```



## How to Optimize the Model

write para_space in flags.py and run `python main.py --mode opt`

```python
para_space = { 
   "space_vec": ['tfidf'],  ## optional vector types ['tfidf', 'count']
   "space_model": ['nmf', 'lsi', 'lda'], ## optional model type ['nmf', 'lsi', 'lda']
   "space_num_topics": range(10,11), ## opctional number of topics
   "space_max_features": [3000]	## optional max features in feature vector
}
```



## How to Run as App

prerequisite: train and save the corresponding model in folder ./save

run `python app.py` to start the local server and request

call `POST ./predict --data '{"text": ["...", "..."]}'` to predict text topics and related docs in training

call `GET ./topic` to get topic list of the trained model

check ./demo for sample code of request



## Requirements

run `pip install -r requirements.txt` to install related packages

