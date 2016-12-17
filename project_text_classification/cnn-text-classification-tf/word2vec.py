import gensim

class Word2vec(object):

	def __init__(self, path_='../tweetdatabase_word2vec'):
		print("\nloading word2vec database")
		try:
			self.model = gensim.models.Word2Vec.load_word2vec_format(path_, binary=True)
		except UnicodeDecodeError as e:
			print(e)
			self.model = gensim.models.Word2Vec.load(path_)
		self.model.init_sims(replace=True)
		print("\nloading done !")

	def isin(self, word):
		return (word in self.model.vocab)

	def get(self, word):
		if self.isin(word):
			return self.model[word]


if __name__ == '__main__':
	w2v = Word2vec()
	while True:
		input_ = input("=>")
		print(input_ , ": ", w2v.isin(input_) ,"\n")

		if w2v.isin(input_):
			print(w2v.get(input_))