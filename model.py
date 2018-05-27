from LSTM import LSTM
from data.myparams import *
from data.datafeeder import DataFeeder

import os
import tensorflow as tf
import numpy as np


if __name__=='__main__':
    with tf.Session() as sess:

        datapath = 'data/Categories/'
        batch_size = params_rnn.batch_size
        train_writer = tf.summary.FileWriter('./NCLS_log/train',sess.graph)
        test_writer = tf.summary.FileWriter('./NCLS_log/test',sess.graph)

        data = DataFeeder(datapath=datapath,
                       bank_size=params_rnn.bank_size,
                       split_rate=params_rnn.split_rate,
                       max_len_sen=params_share.max_len_sen,
                       max_len_news=params_share.max_len_news,
                       embedding_size=params_share.embedding_size,
                       endtoken="endtoken")

        model = LSTM(rnn_size = params_rnn.rnn_size,
                    learning_rate=params_rnn.learning_rate,
                    layer_depth = params_rnn.layer_depth,
                    batch_size=params_rnn.batch_size,
                    embedding_size=params_share.embedding_size,
                    max_len_sen=params_share.max_len_sen,
                    max_len_news=params_share.max_len_news,
                    max_pool_size=params_rnn.max_pool_size)

        sess.run(tf.global_variables_initializer())

        train_merge_step=0

        for i in range(params_rnn.train_epoch):
            ix = len(data.y_train)
            idx = np.random.choice(ix,ix,replace=False)
            loss=0.0
            count=0
            acc=0.0

            for step in range(len(idx)//params_rnn.batch_size):

                ix = idx[step*batch_size:(step+1)*batch_size]

                x,y,x_len = data.next_batch(data.x_train,data.y_train,ix)

                feed_dict={model.input_embed:x,
                           model.input_y:y,
                           model.seqlen:x_len,
                           model.phase:True}

                l1,_,acc1,train_merge=sess.run([model.loss,
                                                model.train_step,
                                                model.accuracy,
                                                model.merged],
                                               feed_dict=feed_dict)

                train_writer.add_summary(train_merge,train_merge_step)
                train_merge_step+=1
                count+=1
                loss+=l1
                acc+=acc1


            x,y,x_len = data.next_batch(data.x_train,data.y_train,
                                    list(range(len(data.y_test))))

            feed_dict={model.input_embed:x,
                       model.input_y:y,
                       model.seqlen:x_len,
                       model.phase:False}

            l2,acc2,test_merge = sess.run([model.loss,
                                           model.accuracy,
                                           model.merged],
                                          feed_dict=feed_dict) 

            test_writer.add_summary(test_merge,i)

            print("[%d][train_loss]%0.4f [train_accuracy]%0.4f [test_loss]%0.4f [test_accuracy]%0.4f"%(i,loss/count,acc/count,l2,acc2))
