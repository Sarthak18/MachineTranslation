import numpy;
import random;
import math;
import gensim;
import pandas as pd;
import cPickle
import gzip

def load(file_name):
    # load the model
    stream = gzip.open(file_name, "rb")
    model = cPickle.load(stream)
    stream.close()
    return model


def save(file_name, model):
    # save the model
    stream = gzip.open(file_name, "wb")
    cPickle.dump(model, stream)
    stream.close()

#eng_model = gensim.models.Word2Vec.load("fname.txt");
eng_model = gensim.models.Word2Vec.load("en_wv");
eng_file = open("en", "r")
eng_lines = eng_file.readlines();

sp_model = gensim.models.Word2Vec.load("sp_wv");
sp_file = open("es", "r")
sp_lines = sp_file.readlines();



#d = 100 and h = 100
m = math.sqrt(6.0/(100+300));
W = numpy.random.uniform(-m,m,[100,100]);
Whh = numpy.random.uniform(-m,m,[100,100]);
B = numpy.random.uniform(-m,m,[100,1]);
Hin = numpy.zeros([100,1]);
Hout = numpy.zeros([100,1]);

n = math.sqrt(6.0/(len(eng_model.wv.vocab)+200));
Ws = numpy.random.uniform(-n,n,[len(sp_model.wv.vocab),100]);
Bs = numpy.random.uniform(-n,n,[len(sp_model.wv.vocab),1]);




for h in range(0,2):
	print(h);
	for row in range(0,len(eng_lines)):
		print(row);
		words = eng_lines[row].strip('\r\n').split(' ');
		HinArr = [];
		HoutArr = [];
		xArr = [];
		ind = 0;
		dBsArr = [];
		dWsArr = [];
		sumdW = numpy.zeros([100,100]);
		sumdB = numpy.zeros([100,1]);
		sumdWhh = numpy.zeros([100,100]);
		sumdBs = numpy.zeros([len(sp_model.wv.vocab),1]);
		sumdWs = numpy.zeros([len(sp_model.wv.vocab),100]);

		for i in range(0, len(words)):
			HinArr.insert(i,Hin);
			word = words[i];
			X = eng_model.wv[word];
			X1 = X.reshape(100,1)
			xArr.insert(i,X1);
		
			Ho = numpy.dot(W,X1) + numpy.dot(Whh,Hin) + B;
			Hout = 1.0 / (1.0 + numpy.exp(-1.0 * Ho))

			Hin = Hout;
			HoutArr.insert(i,Hin);		
			ind = i;

		sp_words = sp_lines[row].strip('\r\n').split(' ');
		for j in range(0, len(sp_words)):
			HinArr.insert(ind+j,Hin);

			sp_j_word = sp_words[j];
			if(j==0):
				X1 = xArr[0];
			else:
				sp_word = sp_words[j-1];
				X = sp_model.wv[sp_word];
				X1 = X.reshape(100,1)

			xArr.insert(ind+j,X1);

			Ho = numpy.dot(W,X1) + numpy.dot(Whh,Hin) + B;
	  		Hout = 1.0 / (1.0 + numpy.exp(-1.0 * Ho))		
			Hin = Hout;
			HoutArr.insert(ind+j,Hin);
		
			#Softmax
			HsIn = Hout;	
	
			ot = numpy.dot(Ws,HsIn) + Bs;

			soft_ele = numpy.exp(ot - numpy.max(ot));
	   		
			HsOut = soft_ele / soft_ele.sum();

			maxInd = HsOut.argmax(axis=0);
		
			#print(sp_j_word)
		        word_obj = sp_model.wv.vocab[sp_j_word];
			wordInd = word_obj.index;
			origIndexVec = numpy.zeros([len(sp_model.wv.vocab),1]);
			origIndexVec[wordInd] = 1;

			error = origIndexVec - HsOut;
		
			#back propogation
			ita = 0.1;
			dBs = ita*error;
			dBsArr.insert(j,dBs);

			delta = dBs;

			dWs = numpy.dot(numpy.around(delta , decimals=3), numpy.transpose(HsIn));

			dWsArr.insert(j,dWs);
		
			# h*(1-h)
			d = HoutArr[ind+j] * (numpy.ones([100,1]) - HoutArr[ind+j]);
			deL =  numpy.dot(numpy.transpose(Ws),delta) * d;
			dW =   numpy.dot(deL,numpy.transpose(xArr[ind+j]));
			sdW = dW;		

			dB = deL;
			sdB = dB;

			dWhh = numpy.dot(deL,numpy.transpose(HinArr[ind+j]));
			sdWhh = dWhh;

			for k in range(ind+j-1, 1):

				d = HoutArr[k] * (numpy.ones([100,1]) - HoutArr[k]);
				deL =  numpy.dot(numpy.transpose(Whh),deL) * d;
				dW = numpy.dot(deL,numpy.transpose(xArr[k]));
				sdW = sdW + dW;

				dB = deL;
				sdB = sdB + dB;
			
				dWhh = numpy.dot(deL,numpy.transpose(HinArr[k]));
				sdWhh = sdWhh + dWhh;	
	
			#for each word summing all the del Weights
			sumdW = sumdW + sdW;
			sumdB = sumdB + sdB;
			sumdWhh = sumdWhh + sdWhh;

			sumdBs = sumdBs + dBs;
			sumdWs += dWs;
	
		#for each sentence updating the weights
		W = W + sumdW;

		B = B + sumdB;
		Whh = Whh + sumdWhh;

		Bs = Bs + sumdBs;
		Ws = Ws + sumdWs;	
		

save("W_Vec",W)
save("Whh_Vec",Whh)
save("Wb_Vec",B)
save("Ws_Vec",Ws)
save("Wsb_Vec",Bs)

print("Trained..!!");

#vocab_obj = model.wv["President"];
#vocab_obj = model.wv.vocab["word"];
#vocab_obj.count;
#vocab_obj.index;

