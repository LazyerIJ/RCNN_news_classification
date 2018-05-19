import re
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request
import data.myutils as myutils
from nltk.corpus import stopwords
from tensorflow.python.framework import ops
from data.myparams import *
from data.datafeeder import DataFeeder

if __name__=='__main__':


    embedding_size = params_share.embedding_size
    vocabulary_size = params_share.vocabulary_size

    batch_size = params_cbow.batch_size
    generations = params_cbow.generations
    model_learning_rate = params_cbow.model_learning_rate
    window_size = params_cbow.window_size
    save_embeddings_every = params_cbow.save_embeddings_every
    print_valid_every = params_cbow.print_valid_every
    print_loss_every = params_cbow.print_loss_every
    num_sampled = params_cbow.num_sampled

    ops.reset_default_graph()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
    ##folder check
    data_folder_name = 'ckpt'
    if not os.path.exists(data_folder_name):
        os.makedirs(data_folder_name)

    ##get data
    #########################################################
    #load
    data = DataFeeder(vocabulary_size=params_share.vocabulary_size,
                   bank_size=params_rnn.bank_size,
                   split_rate=params_rnn.split_rate,
                   max_len_sen=params_share.max_len_sen,
                   processes=True,
                   endtoken="endtoken")
    #process
    texts = data.data_x

    #make dictionary
    word_dictionary = data.word_dict
    word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
    #text to num
    #text_data = utils.text_to_numbers(texts, word_dictionary)
    text_data = data.data_x_number
    #########################################################

    ##valid
    valid_words = ['money','phone','america','asia']
    valid_examples = [word_dictionary[x] for x in valid_words]    
    

    ##model
    sess = tf.Session()
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                   stddev=1.0 / np.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    x_inputs = tf.placeholder(tf.int32, shape=[batch_size, 2*window_size])
    y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    embed = tf.zeros([batch_size, embedding_size])
    for element in range(2*window_size):
        embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=y_target,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))
                                         
    optimizer = tf.train.AdamOptimizer(learning_rate=model_learning_rate).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    saver = tf.train.Saver({"embeddings": embeddings})

    init = tf.global_variables_initializer()
    sess.run(init)

    text_data = [x for x in text_data if len(x)>=(2*window_size+1)]

    print('Starting Training')
    loss_vec = []
    loss_x_vec = []
    for i in range(generations):
        batch_inputs, batch_labels = myutils.generate_batch_data(text_data, batch_size,
                                                                      window_size, method='cbow')
        feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}

        # Run the train step
        sess.run(optimizer, feed_dict=feed_dict)

        # Return the loss
        if (i+1) % print_loss_every == 0:
            loss_val = sess.run(loss, feed_dict=feed_dict)
            loss_vec.append(loss_val)
            loss_x_vec.append(i+1)
            print('Loss at step {} : {}'.format(i+1, loss_val))
          
        # Validation: Print some random words and top 5 related words
        if (i+1) % print_valid_every == 0:
            sim = sess.run(similarity, feed_dict=feed_dict)
            for j in range(len(valid_words)):
                valid_word = word_dictionary_rev[valid_examples[j]]
                top_k = 15 # number of nearest neighbors
                nearest = (-sim[j, :]).argsort()[1:top_k+1]
                log_str = "Nearest to {}:".format(valid_word)
                for k in range(top_k):
                    close_word = word_dictionary_rev[nearest[k]]
                    log_str = '{} {},' .format(log_str, close_word)
                print(log_str)
                
        # Save dictionary + embeddings
        if (i+1) % save_embeddings_every == 0:
            # Save vocabulary dictionary
            with open(os.path.join(data_folder_name,'movie_vocab.pkl'), 'wb') as f:
                pickle.dump(word_dictionary, f)
            
            # Save embeddings
            model_checkpoint_path = os.path.join(os.getcwd(),data_folder_name,'cbow_movie_embeddings.ckpt')
            save_path = saver.save(sess, model_checkpoint_path)
    print('Model saved in file: {}'.format(save_path))
