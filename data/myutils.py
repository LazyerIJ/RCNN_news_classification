import pickle
import re
#from nltk.corpus import stopwords
#from textblob import Word
import collections
from nltk.stem import PorterStemmer
import tarfile
import requests
import numpy as np
import os
import csv
import sys
import string
from zipfile import ZipFile
import nltk
import itertools

maxInt = sys.maxsize
decrement = True
fname = 'ckpt/processed_data.pkl'

while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

def clean_str(string):

    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`.]", " ", string)#<inju>except"."
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\."," endtoken ",string)#<inju>replace endtoken
    return string.strip().lower()


def cleaning_data(texts):
    print('[*]run cleaning_data..')

    if os.path.isfile(fname):
        print('[*]get processed_data..')
        texts = pickle.load(open(fname,'rb'))

    else:
        print('[*]set processed_data..')
        stops = stopwords.words('english')
        st = PorterStemmer()
        for index,value in enumerate(texts):
            texts[index] = ' '.join([st.stem(w) for w in clean_str(value).split()])
            texts[index] = ' '.join([Word(word).lemmatize() for word in texts[index].split()])
        texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]
        texts = [' '.join(x.split()) for x in texts]

        with open(os.path.join(fname),'wb') as f:
            pickle.dump(texts,f)
        f.close()

    return texts

def load_data(datapath):
    import os
    files = []
    x=[]
    y=[]

    for f in os.listdir(datapath):
        if f.endswith(".csv"):
            files.append(f)

    for f in files:
        text_data=[]
        with open(datapath+'/'+f,'r') as f:
            print('[*]read dataset.csv : ' , f)
            reader = csv.reader(f)

            for row in reader:
                text_data.append(row)
            text_data=text_data[1:]
        
        x += [A[0] for A in text_data]
        y += [A[1] for A in text_data]


    return x,y

def split_max_length(data_x,max_sen_len=35,token="endtoken"):
    for i,data in enumerate(data_x):
        data_x[i] = ''.join([x+token for x in data.split(token)[:max_sen_len]])
    return data_x


def split_min_word(data_x,min_word_num,endtoken):
    for i,data in enumerate(data_x):
        data_x[i] = ''.join([x+endtoken for x in data.split(endtoken)
                             if len(x) > min_word_num])
    return data_x

def split_list(iters, values):
    return [list(g) for k,g in itertools.groupby(iters,lambda x:x in values) if not k]

def padding_sentence(news, max_len_sen,embedding_size):

    for i,sentence in enumerate(news):
        cur_len_sen=len(sentence)

        if cur_len_sen>=max_len_sen:
            news[i]=news[i][:max_len_sen]
        else:
            for step in range(max_len_sen-cur_len_sen):
                news[i].append(np.zeros(embedding_size))
    
    return news
                

def padding_news(data,embedding_size,max_len_news,max_len_sen):
    cur_len_news = len(data)
    pad_arr = np.zeros((max_len_sen,embedding_size))
    if cur_len_news>=max_len_news:
        data = data[:max_len_news]
    else:
        for step in range(max_len_news-len(data)):
            data.append(pad_arr)
    return data

