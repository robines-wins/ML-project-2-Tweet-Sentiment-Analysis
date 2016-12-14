import gensim

model = gensim.models.Word2Vec.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)

while True:
	input_ = input()
	print(input_ +": " (input_ in model.vocab)+"\n")