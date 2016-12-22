import gensim

class Word2vec(object):
	"""
	Gensim word2vec model wrapper. load the data and allow us to query it.
	"""

	def __init__(self, path_='../tweetdatabase_word2vec'):
		print("\nloading word2vec database")
		try: #suppose the model is from the C tools instead of python one
			self.model = gensim.models.Word2Vec.load_word2vec_format(path_, binary=True)
		except UnicodeDecodeError as e: #if error then the W2v as been created using python, and need to be load a different way
			self.model = gensim.models.Word2Vec.load(path_)
		self.model.init_sims(replace=True) #allow only read only operation on the model, use fewer ram 
		print("\nloading done !")

	def isin(self, word):
		"""
		Checks the existence of a word in the database

		IN:
		word :		the word for which we want to check existence in the database

		"""
		return (word in self.model.vocab)

	def get(self, word): 
		"""
		Returns the vector representation of a given word if exist in the model
		
		IN:
		word :		the word for which we want to find a representation if it exists
		"""
		if self.isin(word):
			return self.model[word]


if __name__ == '__main__': #If launched as a script, setup a query tool on model at default path
	w2v = Word2vec()
	while True:
		input_ = input("=>")
		print(input_ , ": ", w2v.isin(input_) ,"\n")

		if w2v.isin(input_):
			print(w2v.get(input_))