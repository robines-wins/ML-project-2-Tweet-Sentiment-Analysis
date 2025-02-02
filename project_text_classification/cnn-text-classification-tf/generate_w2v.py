import gensim
import sys

class Sentences(object):
	"""A memeory friendly iterable object which iterates over all words in its list of text files"""

	def __init__(self, filel):
		"""
		Classical python initializer

		IN:
		filel: 	list of path to all text file to iterate on
		"""
		self.filel = filel
 
	def __iter__(self):
		for fname in self.filel: #for each file
			progress = []
			count = 0
			for line in open(fname,'r'): #for each line
				count += 1
				if(count % 100000 == 0):
					progress.append(".")
					toPrint = ''.join(progress)
					sys.stdout.write(toPrint)
					sys.stdout.flush()
				yield line.split() #yield each word
			sys.stdout.write('\n')

def generate_word2vec(filelist,outputfile,dim = 128):
	"""
	Generates a gensim word2vec model form words in several text files
	
	IN : 
	filelist: 		List of path to all file used to generate the model
	outputefile: 	path to the file where the model will be saved for later use
	dim: 			dimension of the vectors in the generated model (to be chosen, else will be set to 128 by default) 

	"""
	print("Generating word2vec files (This might take a while) ...")
	sentences = Sentences(filelist)
	model = gensim.models.Word2Vec(sentences,workers=4,size=dim)
	model.init_sims(replace=True) #pass the model in read-only, use fewer memory
	model.save(outputfile)
	print("Done ! ")

if __name__ == "__main__": #if called as a script, will generate a model from all our dataset
	filelist=['../twitter-datasets/train_pos_full.txt','../twitter-datasets/train_neg_full.txt','../twitter-datasets/test_data.txt']
	outputfile = '../tweetdatabase_word2vec'
	generate_word2vec(filelist,outputfile)