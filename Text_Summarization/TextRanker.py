import _pytextrank as PTR
import spacy
import zipfile
import json
import time

# import networkx as nx
# import pylab as plt

def read_text_zip(zip_loc ):
	print("reading data")
	zfile = zipfile.ZipFile(zip_loc)
	re = []
	global metadata
	metadata = {}
	ind = 0
	for finfo in zfile.infolist():
		ifile = zfile.open(finfo, 'r')
		fname = finfo.filename
		metadata[ind] = { "fname": fname }
		ind += 1
		doc = ifile.read().decode("utf-8")
		re.append(doc)
	return re, metadata

class TextRanker():
	def __init__(self, input_data = None):
		self.stopwords = "stop.txt"
		self.spacy_nlp = None
		self.skip_ner = True
		self.phrase_limit = 15
		self.sent_word_limit = 150

		if input_data:
			self.__input_data = input_data

	def set_data(self, input_data):
		self.__input_data = input_data

	def get_data(self):
		return self.__input_data

	def run(self):
		# Parse text
		parse = PTR.parse_doc(PTR.text2json(self.__input_data))
		parse_list = [json.loads(PTR.pretty_print(i._asdict())) for i in parse]
		# print(parse_list[0]['graf'])
		
		# Create and rank graph for keywords
		graph, ranks = PTR.text_rank(parse_list)
		norm_rank = PTR.normalize_key_phrases(parse_list, ranks, 
											stopwords = self.stopwords, 
											spacy_nlp = self.spacy_nlp, 
											skip_ner = self.skip_ner
											)
		
		norm_rank_list = [json.loads(PTR.pretty_print(rl._asdict())) for rl in norm_rank ]
		# print(norm_rank_list)

		keywords = set([p for p in PTR.limit_keyphrases(norm_rank_list, 
														phrase_limit = self.phrase_limit
														)])

		# return a matrix like result for the top keywords
		kernel = PTR.rank_kernel(norm_rank_list)

		# Rank the sentences
		top_sent = PTR.top_sentences(kernel, parse_list)
		top_sent_list = [json.loads(PTR.pretty_print(s._asdict())) for s in top_sent ]
		sent_iter = sorted(PTR.limit_sentences(top_sent_list, 
												word_limit = self.sent_word_limit
												), key=lambda x: x[1])

		# Return ranked sentences
		sentences = []
		for sent_text, idx in sent_iter:
			sentences.append(PTR.make_sentence(sent_text))

		summary = " ".join(sentences)

		return summary, keywords

if __name__ == '__main__':

	data, metadata = read_text_zip('../shared-data/wikiLarge.zip')

	input_data = data[0]
	print(metadata[0])

	TR = TextRanker(input_data)
	
	t0 = time.time()
	s, w = TR.run()
	print("summary:", s)
	print("keywords:", w)
	print(time.time() - t0)

