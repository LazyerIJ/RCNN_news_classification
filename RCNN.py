import os
import tensorflow as tf
import numpy as np
from data.myparams import *
from data.datafeeder import DataFeeder


class LSTM():

    def __init__(self,learning_rate,rnn_size,layer_depth,batch_size,embedding_size,max_len_news,max_len_sen,max_pool_size):

        self.learning_rate=learning_rate
        self.rnn_size=rnn_size
        self.batch_size=batch_size
        self.layer_depth=layer_depth
        self.embedding_size=embedding_size
        self.max_len_news=max_len_news
        self.max_len_sen=max_len_sen
        cell = []

        self.input_y = tf.placeholder(tf.float32,shape=[None,params_share.label_num],name='x')
        self.input_embed = tf.placeholder(tf.float32, shape=[None,self.max_len_news,self.max_len_sen,self.embedding_size],name='embeded')
        self.seqlen= tf.placeholder(tf.int32, shape=[None],name='len')
        self.phase = tf.placeholder(tf.bool, name='phase')

        def dense_batch_relu(x,phase,scope):
            with tf.variable_scope(scope):
                h1 = tf.contrib.layers.fully_connected(x,32,scope='dense')
                h2 = tf.contrib.layers.batch_norm(h1,center=True,scale=True,
                                                  is_training=phase,scope='bn')
                return tf.nn.relu(h2,'relu')

        def dense(x,size,scope):
            return tf.contrib.layers.fully_connected(x,size, scope=scope)

        for i in range(params_rnn.layer_depth):
            cell.append(tf.contrib.rnn.LSTMCell(params_rnn.rnn_size, 
                                                state_is_tuple=True))
        self.stacked_cell = tf.contrib.rnn.MultiRNNCell(cell,
                                                        state_is_tuple=True)

        result=[]
        for i, filter_size in enumerate(params_rnn.bank_size):

            with tf.variable_scope('conv_%s'%i):

                filter_shape = [1,
                                filter_size,
                                self.embedding_size,
                                params_rnn.num_filters]

                W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W1')
                b = tf.Variable(tf.constant(0.0,shape=[params_rnn.num_filters]),name='b1')
                conv =tf.nn.conv2d(self.input_embed,W,strides=[1,1,1,1],padding='VALID',name='conv2d1')##insert
                conv_out = tf.nn.sigmoid(tf.nn.bias_add(conv,b),name='sigmoid1')
                h = tf.reshape(conv_out,[-1,params_share.max_len_news,(params_share.max_len_sen-filter_size+1)*params_rnn.num_filters,1])

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1,1,max_pool_size,1],
                    strides=[1,1,max_pool_size,1],
                    padding='VALID',
                    name='pool')
                print('[cnn]pooled : ' , pooled)
            result.append(pooled)

        result = tf.concat(result,2)
        print('[cnn]cnn result : ', result)
        cnn_output = tf.squeeze(result,3)

        self.outputs, _ = tf.nn.dynamic_rnn(
            self.stacked_cell,
            cnn_output,
            dtype=tf.float32,
            sequence_length=self.seqlen)
        
        print('[*]outputs : ' , self.outputs) 

        self.outputs = tf.reshape(self.outputs,[self.max_len_news,-1,params_rnn.rnn_size])[-1]
        print('[*]reshape outputs : ' , self.outputs) 

        h1 = dense_batch_relu(self.outputs,self.phase,'layer1')
        h2 = dense_batch_relu(h1,self.phase,'layer2')
        self.logits = dense(h2,params_share.label_num,'logits')
        print('[*]lotis : ' , self.logits)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.input_y))

        self.merged = tf.summary.merge_all()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.GradientDescentOptimizer(params_rnn.learning_rate).minimize(self.loss)

        #self.train_step = tf.train.GradientDescentOptimizer(params_rnn.learning_rate).minimize(self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.input_y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float32"))

        tf.summary.scalar("loss",self.loss)
        tf.summary.scalar("accuracy",self.accuracy)
        self.merged = tf.summary.merge_all()



