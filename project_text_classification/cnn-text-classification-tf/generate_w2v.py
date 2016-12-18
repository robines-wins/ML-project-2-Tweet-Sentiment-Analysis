import gensim

class Sentences(object):
	"""an memeory friendly iterable object who iterate over all words in its list of text files"""
	def __init__(self, filel):
		"""IN:
		filel: list of path to all text file to iterate on
		"""
		self.filel = filel
 
	def __iter__(self):
		for fname in self.filel: #for each file
			for line in open(fname,'r'): #for each line
				yield line.split() #yield each word

def generate_word2vec(filelist,outputfile,dim = 128):
	"""
	generate a gensim word2vec model form words in serval text files
	IN : 
	filelist: List of path to all file used to generate the model
	outputefile: path to the file where the model will be saved for later use
	dim: dimension of the vectors in the generates model 

	"""
	sentences = Sentences(filelist)
	model = gensim.models.Word2Vec(sentences,workers=4,size=dim)
	model.init_sims(replace=True)
	model.save(outputfile)

if __name__ == "__main__": #if call as script will generate a model from all our dataset
	filelist=['../twitter-datasets/train_pos_full.txt','../twitter-datasets/train_neg_full.txt','../twitter-datasets/test_data.txt']
	outputfile = '../tweetdatabase_word2vec'
	generate_word2vec(filelist,outputfile)