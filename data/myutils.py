import pickle
import re
from nltk.corpus import stopwords
from textblob import Word
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

def load_data():

    print('[*]run load_data..')
    data_x=[]
    data_y=[]
    text_data=[]

    if os.path.isfile('data/dataset_processed.csv'):
        with open('data/dataset_processed.csv','r') as f:
            print('[*]read processed.csv')
            reader = csv.reader(f)
            for row in reader:
                text_data.append(row)
        data_x=text_data[0]
        data_y=text_data[1]

    else:
        with open('data/dataset.csv','r') as f:
            print('[*]read dataset.csv')
            reader = csv.reader(f)
            for row in reader:
                text_data.append(row)
            text_data=text_data[1:]
        data_x = [A[0] for A in text_data]
        data_y = [A[1] for A in text_data]

    return data_x,data_y

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
    fname = 'ckpt/processed_data.pkl'

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


def build_dictionary(sentences, vocabulary_size):
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]
    count= [['RARE',-1]]

    count.extend(collections.Counter(words).most_common(vocabulary_size-1))

    word_dict = {}
    for word, word_count in count:
        word_dict[word] = len(word_dict)

    return word_dict


def text_to_numbers(sentences, word_dict):
    data=[]
    for sentence in sentences:
        sentence_data = []
        for word in sentence.split(' '):
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix=0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return (data)

def split_max_length(data_x,data_y,max_sen_len=35,token="endtoken"):

    data_x_split = [x for x in data_x 
                    if len(x.split(token))<=max_sen_len]
    data_y_split = [data_y[i] for i,ix in enumerate(data_x)
                    if len(ix.split(token))<=max_sen_len]
    return data_x_split, data_y_split

def split_min_word(data_x,min_word_num,endtoken):
    for i,data in enumerate(data_x):
        data_x[i] = ''.join([x+endtoken for x in data.split(endtoken)
                             if len(x) > min_word_num])
    return data_x

def split_list(iters, values):
    return [list(g) for k,g in itertools.groupby(iters,lambda x:x in values) if not k]

def padding_data(data, seq_length):

    for i,news in enumerate(data):
        for j,sentence in enumerate(news):
            sen_len=len(sentence)
            if sen_len>seq_length:
                data[i][j]=data[i][j][:seq_length]
            else:
                data[i][j]=np.pad(data[i][j],(0,seq_length-sen_len),'constant',
                                  constant_values=(0))
    return data
                

def padding_sentence_length(data,seq_length,seq_arr):
    pad_arr = np.zeros((seq_length))

    for i,news in enumerate(data):
        cur_len = seq_arr[i]
        add_size = seq_length - cur_len
        for step in range(add_size):
            data[i].append(pad_arr)

    return data

def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    batch_data=[]
    label_data=[]

    while len(batch_data) < batch_size:

        rand_sentence = np.random.choice(sentences)
        window_sequences = [rand_sentence[max((ix-window_size),0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]

        label_indices = [ix if ix<window_size else window_size for ix, x in enumerate(window_sequences)]

        if method=='skip_gram':

            batch_and_labels=[(x[y],x[:y]+x[(y+1):]) 
                              for x,y 
                              in zip(window_sequences,label_indices)]

            tuple_data = [(x,y_) for x,y in batch_and_labels 
                          for y_ in y]

            batch,labels = [list(x) for x in zip(*tuple_data)]

        elif method=='cbow':
            batch_and_labels= [(x[:y]+x[(y+1):], x[y]) for x,y in zip(window_sequences,
                                                                      label_indices)]
            batch_and_labels = [(x,y) for x,y in batch_and_labels if len(x)==2*window_size]
            batch,labels = [list(x) for x in zip(*batch_and_labels)]

        else:
            raise ValueError('Method P{ not implemented yet.'.format(method))

        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])

    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return (batch_data, label_data)

