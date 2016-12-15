import word2vec
import numpy as np
#import cPickle as pickle

class Vocabulary(object):

	def __init__(self,maxlength, w2v=None,vocabulary=None,dim=300):
		self.maxlength = maxlength
		self.w2v = w2v
		self.vocsize = 1
		self.vocab = {}
		self.embeding = [np.zeros(dim)]
		self.dim=dim


	def fit(self,strlist):
		for s in strlist:
			for w in s.split():
				if w not in self.vocab:
					self.vocab[w]=self.vocsize
					self.vocsize += 1
					if self.w2v != None and self.w2v.isin(w):
						self.embeding.append(w2v.get(w))
					else:
						self.embeding.append(np.random.uniform(low=-1.0,high=1.0,size=self.dim))

	def transform(self,strlist):
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
		self.fit(strlist)
		return self.transform(strlist)

	def embedingMatrix(self):
		return np.array(self.embeding)

	#def __getstate__(self):
