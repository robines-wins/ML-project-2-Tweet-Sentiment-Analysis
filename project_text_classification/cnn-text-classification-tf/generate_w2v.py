import gensim

class Sentences(object):
	def __init__(self, filel):
		self.filel = filel
 
	def __iter__(self):
		i = 0
		for fname in self.filel:
			for line in open(fname,'r'):
				i += 1
				if i%10000 == 0: print(i)
				yield line.split()

def generate_word2vec(filelist,outputfile,dim = 128):
	sentences = Sentences(filelist)
	model = gensim.models.Word2Vec(sentences,workers=4,size=dim)
	model.init_sims(replace=True)
	model.save(outputfile)

if __name__ == "__main__":
	filelist=['../twitter-datasets/train_pos_full.txt','../twitter-datasets/train_neg_full.txt']
	outputfile = '../tweetdatabase_word2vec'
	generate_word2vec(filelist,outputfile)