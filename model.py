from LSTM import LSTM
from data.myparams import *
from data.datafeeder import DataFeeder

import os
import tensorflow as tf
import numpy as np


if __name__=='__main__':

    with tf.Session() as sess:

        data_folder_name='ckpt'

        data = DataFeeder(vocabulary_size=params_share.vocabulary_size,
                   bank_size=params_rnn.bank_size,
                   split_rate=params_rnn.split_rate,
                   max_len_sen=params_share.max_len_sen,
                   processes=True,
                   endtoken="endtoken")

        data.process_data(max_len_word=params_share.max_len_word)

        test = LSTM(rnn_size = params_rnn.rnn_size,
                    learning_rate=params_rnn.learning_rate,
                    layer_depth = params_rnn.layer_depth,
                    batch_size=params_rnn.batch_size,
                    vocabulary_size=params_share.vocabulary_size,
                    embedding_size=params_share.embedding_size,
                    max_len_sen=params_share.max_len_sen,
                    max_len_word=params_share.max_len_word)
        print('[*]vocabulary_size : ' , test.vocabulary_size)
        print('[*]embedding  size : ' , test.embedding_size)

        writer = tf.summary.FileWriter('./board/train', sess.graph)


        sess.run(tf.global_variables_initializer())

        # Load model embeddings
        model_checkpoint_path = os.path.join(data_folder_name,'cbow_movie_embeddings.ckpt')
        saver = tf.train.Saver({"embeddings": test.embeddings})
        saver.restore(sess, model_checkpoint_path)
        i=0
        for step in range(params_rnn.train_epoch):
            loss=0.0 
            count=1
            a=[]
            b=[]
            for a,b,c in data.next_batch(params_rnn.batch_size):
                feed_dict={test.input_x:a}
                input_embed = sess.run(test.input_embeded,feed_dict=feed_dict)
                feed_dict={test.input_embed:input_embed,test.input_y:b,test.seqlen:c,test.phase:True}
                l,_,=sess.run([test.loss,test.train_step],feed_dict=feed_dict)

                #writer.add_summary(summ,i) 
            feed_dict={test.input_x:data.x_train}
            input_embed = sess.run(test.input_embeded,feed_dict=feed_dict)

            feed_dict={test.input_embed:input_embed,test.input_y:data.y_train,test.seqlen:data.len_train,test.phase:True}
            l1,acc1,summ = sess.run([test.loss,test.accuracy,test.merged],feed_dict=feed_dict) 
            writer.add_summary(summ,i) 
            i+=1
            feed_dict={test.input_x:data.x_test}
            input_embed = sess.run(test.input_embeded,feed_dict=feed_dict)

            feed_dict={test.input_embed:input_embed,test.input_y:data.y_test,test.seqlen:data.len_test,test.phase:False}
            l2,acc2 = sess.run([test.loss,test.accuracy],feed_dict=feed_dict) 
            
            print("[%d][train_loss]%0.4f [train_accuracy]%0.4f [test_loss]%0.4f [test_accuracy]%0.4f"%(step,l1,acc1,l2,acc2))
        print("finished")

