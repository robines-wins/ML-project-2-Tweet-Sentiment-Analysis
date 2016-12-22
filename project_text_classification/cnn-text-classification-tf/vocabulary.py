import word2vec
import numpy as np
import pickle

class Vocabulary(object):
	"""
	A vocabulary class used to transform strings into lists of indices
	while maintaining an embedding matrix of the vector representation of 
	each word/indices
	Word representation are taken from a given Word2vec model.
	if no such model is provided all vector will be gererated randomly
	"""

	def __init__(self,maxlength, w2v=None,dim=128):
		"""
		Classic python initializer to construct the vocabulary object

		IN :
		maxlength :		the maximum length in terms of words of our sentences
		w2v : 			pretrained word2vectors wrapper (Default : None)
		dim : 			size of the vectors corresponding to each word
		"""
		self.maxlength = maxlength
		self.w2v = w2v
		self.vocsize = 1
		self.vocab = {}
		self.embedding = [np.zeros(dim)] #if word unknown than id will be 0 and associate vector the numm vector
		self.dim=dim


	def fit(self,strlist):
		"""
		Builds the vocabulariy from strlist. This will be done in two possible ways. 

		1. if a pretrained vector set w2v was passed in the constructor then it will use it for each word to check 
		if it is already in this set.
			- if it is in this set, it will assign to a given word the pretrained vector
			- otherwise it will assign a vector randomly generated.
		2. if no pretrained vector set was passed then all words will be assigned to a randomly distributed vector.

		IN : 
		strlist:	the list of of senetences for which we want to build a vocabulary
		"""

		for s in strlist:
			for w in s.split():
				if w not in self.vocab:
					self.vocab[w]=self.vocsize
					self.vocsize += 1
					if self.w2v != None and self.w2v.isin(w):	
						#print(w, " is in w2v")
						self.embedding.append(self.w2v.get(w))
					else:
						self.embedding.append(np.random.uniform(low=-1.0,high=1.0,size=self.dim))

	def transform(self,strlist):
		"""
		Will use the vocabulary built by fit(.) to transform the list of tweets into a list of 
		vectors. Each vector corresponds to a sentence and contains the index in the embedding 
		of the given word (obtained from the vocabulary.

		IN : 
		strlist:	the list of of senetences for which we want to build a vocabulary

		OUT:
		list of vectors of same size than the number of tweets
		"""

		tr = []
		for s in strlist:
			vec = np.zeros(self.maxlength,np.int64)
			for idx,w in enumerate(s.split()):
				if idx >= self.maxlength:
					break
				if w in self.vocab:
					vec[idx] = self.vocab[w]
			tr.append(vec)
		return tr

	def fit_transform(self,strlist):
		"""
		Will first build the vocabulary and then return the list of vectors corresponding to each sentences

		IN : 
		strlist:	the list of of senetences for which we want to build a vocabulary

		OUT:
		list of vectors of same size than the number of tweets
		"""

		self.fit(strlist)
		return self.transform(strlist)

	def embeddingMatrix(self):
		"""
		Transforms the embeding list into a np array

		OUT:
		The embeding matrix
		"""

		return np.array(self.embedding, dtype='float32')

	def save(self, filename):
		"""
		Utility method allowing to save the vocabulary so that we can reuse it when evaluating the model on new inputs

		IN:
		self :		the vocabulary object
		filename :	the name of the file to which we wish to save our vocabulary
		"""
		f = open(filename,'wb')
		pickle.dump(self,f,-1)
		f.close()

	@classmethod
	def restore(cls, filename, w2v = None):
		"""
		Utility method allowing use to restore the vocabulary to reuse it on new inputs for the same model

		IN:
		filename:	the name of the file from which we wish to restore our vocabulary
		w2v:		pretrained word2vectors model (Default : None)
		"""
		f = open(filename,'rb')
		voc = pickle.load(f)
		f.close()
		voc.w2v = w2v
		return voc

	def __getstate__(self):
		tosave = self.__dict__.copy()
		tosave['w2v'] = None
		return tosave