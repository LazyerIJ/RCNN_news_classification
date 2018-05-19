import data.myutils as utils
from data.myparams import *
import numpy as np

class DataFeeder():

    def __init__(self,vocabulary_size,bank_size,split_rate=0.8,max_len_sen=35,processes=True,endtoken="endtoken"):
        
        print("[*]init datafeeder..")
        self.data_x, self.data_y = utils.load_data()

        self.data_x_number=[]
        self.seq_len=[]
        self.x_train=[]
        self.y_train=[]
        self.len_train=[]
        self.x_test=[]
        self.y_test=[]
        self.len_test=[]
        
        self.split_rate=split_rate
        self.vocabulary_size=vocabulary_size
        self.max_len_sen=max_len_sen
        self.endtoken = endtoken

        if processes:
            self.data_x = utils.cleaning_data(self.data_x)

        self.data_x,self.data_y = utils.split_max_length(self.data_x,self.data_y,self.max_len_sen,self.endtoken)

        self.data_x = utils.split_min_word(self.data_x,max(bank_size),self.endtoken)
        self.word_dict = utils.build_dictionary(self.data_x, self.vocabulary_size)

        def onehot(data_y):
            print('[*]make one-hot..')
            y_dict = {}

            for i,ix in enumerate(set(data_y)):
                y_dict.update({ix:i})

            y = np.zeros((len(data_y),len(y_dict)),dtype=np.float32)

            for i,ix in enumerate(y):
                item=data_y[i]
                y[i][y_dict[item]]=1.

            return y

        self.y_onehot = onehot(self.data_y)

        print("text to number..")
        self.data_x_number = utils.text_to_numbers(self.data_x, self.word_dict)

    def process_data(self,max_len_word=50):

        print("[*]process data..")

        #split with endtoken
        print("split with endtoken..")
        self.data_x_number = [utils.split_list(x,(self.word_dict[self.endtoken],)) for x in self.data_x_number]

        print("make sequence length..")
        self.seq_len = [len(x) for x in self.data_x_number]

        print("padding new length..")
        #padding max_len_sen 35 -> 모든 뉴스 문장 개수는  35
        self.data_x_number = utils.padding_sentence_length(self.data_x_number,self.max_len_sen,self.seq_len)

        print("padding word length..")
        #padding max_len_word 50 -> 모든 문장의 단어 개수는 50
        self.data_x_number = utils.padding_data(self.data_x_number,max_len_word)

        print("split train/test set..")
        train_indices = np.random.choice(len(self.data_y), round(len(self.data_y)*self.split_rate), replace=False)
        test_indices = np.array(list(set(range(len(self.data_y))) - set(train_indices)))
        def indices(datas,idxs):
            for idx in idxs:
                for data in datas:
                    yield  np.array([x for ix,x in enumerate(data) if ix in idx])

        self.x_train,self.y_train,self.len_train,self.x_test,self.y_test,self.len_test=indices([self.data_x_number,self.y_onehot,self.seq_len],[train_indices,test_indices])
        print('[*]x_train   : ' , len(self.x_train))
        print('[*]len_train : ' , len(self.len_train))
        print('[*]y_train   : ' , len(self.y_train))
        print('[*]x_test    : ' , len(self.x_test))
        print('[*]len_test  : ' , len(self.len_test))
        print('[*]y_test    : ' , len(self.y_test))


    def next_batch(self,batch_size):
        idx = np.random.choice(len(self.y_train),len(self.y_train),replace=False)
        for step in range(len(idx)//batch_size):
            ix = idx[step*batch_size:(step+1)*batch_size]
            yield self.x_train[ix],self.y_train[ix],self.len_train[ix]
        #index = np.random.randint(len(self.y_train),size=myparams.batch_size)
        #return self.x_train[index], self.y_train[index], self.len_train[index]


if __name__=='__main__':
    t = DataFeeder(vocabulary_size=params_share.vocabulary_size,
                   bank_size=params_rnn.bank_size,
                   split_rate=params_rnn.split_rate,
                   max_len_sen=params_share.max_len_sen,
                   processes=True,
                   endtoken="endtoken")
    t.process_data(max_len_word=params_share.max_len_word)

