import numpy;
import random;
import math;
import gensim;
from gensim.models import KeyedVectors

sentences = gensim.models.word2vec.LineSentence('en')
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
model.save("en_wv")
 

sentences = gensim.models.word2vec.LineSentence('es')
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
model.save("sp_wv")



 








