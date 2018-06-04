import os
import tensorflow as tf
import numpy as np
from data.myparams import *
from data.datafeeder import DataFeeder

class LSTM():

    def __init__(self,learning_rate,sen_rnn_size,news_rnn_size,layer_depth,embedding_size,max_len_news,max_len_sen,filter_sizes,num_filters,attention_size,keep_prob,num_classes,grad_clip):
        
        result=[]

        self.input_embed = tf.placeholder(tf.float32, shape=[None,max_len_news,max_len_sen,embedding_size],name='embeded')
        self.input_y = tf.placeholder(tf.float32,shape=[None,num_classes],name='x')
        self.seqlen= tf.placeholder(tf.int32, shape=[max_len_news,None],name='t2')
        self.newslen= tf.placeholder(tf.int32, shape=[None],name='t1')
        self.global_step = tf.Variable(0,trainable=False,name='global_step')
        
        ###################sen rnn#######################
        with tf.variable_scope('word_rnn'):
            sen_cell=[]
            for i in range(layer_depth):
                sen_cell.append(tf.contrib.rnn.LSTMCell(sen_rnn_size, 
                                                    state_is_tuple=True))
            sen_stacked_cell = tf.contrib.rnn.MultiRNNCell(sen_cell,
                                                            state_is_tuple=True)

            sentence_rnn_result = []
            for step in range(max_len_news):
                rnn_input_by_step = self.input_embed[:,step,:,:]
                outputs,states = tf.nn.dynamic_rnn(
                    sen_stacked_cell,
                    rnn_input_by_step,
                    dtype=tf.float32,
                    sequence_length=self.seqlen[step,:])

                sentence_rnn_result.append(outputs[:,-1,:])

            sentence_rnn_result = tf.stack(sentence_rnn_result)
            sentence_rnn_result = tf.reshape(sentence_rnn_result,
                                             [-1,max_len_news,sen_rnn_size])

            sentence_rnn_result = tf.expand_dims(sentence_rnn_result,-1)

        ###################news cnn#######################
        with tf.variable_scope('sen_cnn'):
            pooled_outputs = []

            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [1, filter_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        sentence_rnn_result,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    print('[*]h : {}'.format(h))
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, 1,sen_rnn_size - filter_size + 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="news_pool")
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1,max_len_news,num_filters_total])

        ###################news rnn#######################

        with tf.variable_scope('sentence_rnn'):
            news_cell=[]
            for i in range(layer_depth):
                news_cell.append(tf.contrib.rnn.LSTMCell(news_rnn_size, 
                                                    state_is_tuple=True))
            news_stacked_cell = tf.contrib.rnn.MultiRNNCell(news_cell,
                                                            state_is_tuple=True)
            rnn_output,states1 = tf.nn.dynamic_rnn(
                news_stacked_cell,
                h_pool_flat,
                dtype=tf.float32,
                sequence_length=self.newslen)

        ####################################################
        with tf.name_scope('attention'):

            input_shape = rnn_output.shape # (batch_size, sequence_length, hidden_size)
            sequence_size = input_shape[1].value  # the length of sequences processed in the RNN layer
            hidden_size = input_shape[2].value  # hidden size of the RNN layer

            attention_w = tf.Variable(tf.truncated_normal([hidden_size, attention_size], stddev=0.1), name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            attention_u = tf.Variable(tf.truncated_normal([attention_size], stddev=0.1), name='attention_u')
            z_list = []
            for t in range(sequence_size):
                u_t = tf.tanh(tf.matmul(rnn_output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
                z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
                z_list.append(z_t)
            # Transform to batch_size * sequence_size
            attention_z = tf.concat(z_list, axis=1)
            self.alpha = tf.nn.softmax(attention_z)
            # Transform to batch_size * sequence_size * 1 , same rank as rnn_output
            attention_output = tf.reduce_sum(rnn_output * tf.reshape(self.alpha, [-1, sequence_size, 1]), 1)

        # Add dropout
        with tf.name_scope('dropout'):
            # attention_output shape: (batch_size, hidden_size)
            self.final_output = tf.nn.dropout(attention_output,keep_prob)

        # Fully connected layer
        with tf.name_scope('output'):
            fc_w = tf.Variable(tf.truncated_normal([hidden_size, num_classes], stddev=0.1), name='fc_w')
            fc_b = tf.Variable(tf.zeros([num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            self.predictions = tf.argmax(self.logits, 1, name='predictions')

        # Calculate cross-entropy loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
        
        # Create optimizer
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)
            #gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            #gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
            #self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        # Calculate accuracy
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if __name__=='__main__':
    t= LSTM(learning_rate=0.01,
            sen_rnn_size=128,
            news_rnn_size=256,
            layer_depth=2,
            batch_size=16,
            embedding_size=100,
            max_len_news=35,
            max_len_sen=50,
            max_pool_size=4,
            filter_sizes=[3,4,5],
            num_filters=32,
            attention_size=100,
            keep_prob=0.5,
            num_classes=5,
            grad_clip=5.0)

