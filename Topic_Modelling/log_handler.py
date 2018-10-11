import logging

def get_logger(name):
	logger = logging.getLogger(name)
	
	logger.setLevel(logging.INFO)

	## file handler
	# fh = logging.FileHandler('spam.log')
	# fh.setLevel(logging.INFO)

	## console handler
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)

	## formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	# fh.setFormatter(formatter)
	
	logger.addHandler(ch)
	# logger.addHandler(fh)
	return logger
