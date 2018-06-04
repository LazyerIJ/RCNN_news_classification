import os
import data.myutils as utils
import gensim
from data.myparams import *
import numpy as np
import pickle

class DataFeeder():

    def __init__(self,datapath,bank_size,split_rate,max_len_news,embedding_size,max_len_sen,endtoken="endtoken"):
        fname = 'processed/gensim_news_wv.ij' 
        processed_name = 'processed/process_data.ij'

        self.seq_len=[]

        self.x_train=[]
        self.y_train=[]
        self.len_train=[]

        self.x_test=[]
        self.y_test=[]
        self.len_test=[]

        self.split_rate=split_rate
        self.max_len_sen=max_len_sen
        self.max_len_news=max_len_news
        self.embedding_size=embedding_size
        self.endtoken = endtoken
        
        ##load_dataset
        ##############################################################
        self.data_x, self.data_y = utils.load_data(datapath)
        print('[*]load data num : ' , len(self.data_y))
        self.data_x = [utils.clean_str(news) for news in self.data_x]

        ##text preprocessing
        ##############################################################
        if os.path.exists(processed_name):
            with open(processed_name, 'rb') as f:
                self.data_x = pickle.load(f)
        else:
            self.data_x = utils.split_max_length(self.data_x,self.max_len_sen,self.endtoken)
            self.data_x = utils.split_min_word(self.data_x,min(bank_size),self.endtoken)
            self.data_x = [gensim.parsing.preprocessing.stem_text(news) for news in self.data_x]
            self.data_x = [gensim.parsing.preprocessing.strip_punctuation2(news) for news in self.data_x]
            self.data_x = [gensim.parsing.preprocessing.strip_numeric(news) for news in self.data_x]
            self.data_x = np.array(self.data_x)

            with open(processed_name, 'wb') as f:
                pickle.dump(self.data_x, f)

        ##onehot
        ##############################################################
        y_dict = {}
        for i,ix in enumerate(set(self.data_y)):
            y_dict.update({ix:i})
        y_t = np.array([y_dict[n] for n in self.data_y])
        self.y_onehot = np.identity(len(y_dict))[y_t]

        ##make word2vec model 
        ##############################################################
        
        if os.path.exists(fname):
            self.model = gensim.models.Word2Vec.load(fname)
        else:
            doct = [gensim.utils.simple_preprocess(line) for line in self.data_x]
            self.model = gensim.models.Word2Vec(min_count=0,size=self.embedding_size,workers=10)
            self.model.build_vocab(doct)
            self.model.train(doct, total_examples=self.model.corpus_count,epochs=50)
            self.model.save(fname)

        ##split data##        
        ##############################################################
        data_num = len(self.data_y)
        train_num = round(data_num*self.split_rate )
        idx = np.random.permutation((data_num))
        self.x_train = self.data_x[idx[:train_num]]
        self.y_train = self.y_onehot[idx[:train_num]]
        self.x_test  = self.data_x[idx[train_num:]]
        self.y_test  = self.y_onehot[idx[train_num:]]

    ##make one news to vec
    ##############################################################
    def news_to_vec(self,news,endtoken):

        news_vec=[] 
        for sen in news.split(endtoken):
            sentences=[]
            for word in sen.split():
                try:
                    word_vec = self.model[word]
                    sentences.append(word_vec)
                except:
                    pass
            news_vec.append(sentences)
        news_len = len(news_vec)
        return news_vec,news_len

    ##make all news to vec
    ##############################################################
    def make_vec_data(self,data):
        vec_data=[]
        vec_data_len=[]
        for news in data:
            news_vec,news_len = self.news_to_vec(news,self.endtoken)
            vec_data.append(news_vec)
            vec_data_len.append(news_len)
        return vec_data,vec_data_len


    def next_batch(self,x,y,ix):
        batch_x = x[ix]
        batch_y = y[ix]
        batch_x_len = [len(news.split("endtoken")) for news in batch_x]

        batch_x_sen = []

        for step in range(self.max_len_news):
            sen_len = []
            for news in batch_x:
                try:
                    sen_len.append(len(news.split("endtoken")[step].split()))
                except:
                    sen_len.append(0)
            batch_x_sen.append(sen_len)


        batch_x_vec, batch_x_len = self.make_vec_data(batch_x) 

        batch_x_vec = [utils.padding_sentence(news,self.max_len_sen,self.embedding_size) for news in batch_x_vec]
    

        batch_x_vec = [utils.padding_news(news,self.embedding_size,self.max_len_news,self.max_len_sen) for news in batch_x_vec]

        return np.array(batch_x_vec),np.array(batch_y),np.array(batch_x_len),np.array(batch_x_sen)


if __name__=='__main__':
    datapath = 'data/Categories'
    t = DataFeeder(datapath=datapath,
                   bank_size=params_rnn.bank_size,
                   split_rate=params_rnn.split_rate,
                   max_len_sen=params_share.max_len_sen,
                   max_len_news=params_share.max_len_news,
                   embedding_size=params_share.embedding_size,
                   endtoken="endtoken")
    for a,b,c in t.next_batch(t.x_train,t.y_train,30):
        print(a.shape)
        print(b.shape)
        print(c)

