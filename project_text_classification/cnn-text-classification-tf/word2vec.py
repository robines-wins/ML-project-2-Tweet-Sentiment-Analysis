import gensim

model = gensim.models.Word2Vec.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
print('computer' in model.vocab)
print('azdazdazdazdazdad' in model.vocab)