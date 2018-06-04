import tensorflow as tf

params_share = ({
    'vocabulary_size' : 1000,
    'embedding_size' : 100,
    'max_len_sen':15,#padding word
    'max_len_news':10,#padding sentence
    'label_num':3
})
params_share = tf.contrib.training.HParams(**params_share)

params_rnn = ({
    'split_rate':0.8,
    'batch_size':3,
    'num_filters':16,
    'bank_size':[3,5],
    'rnn_size':256,
    'layer_depth':3,
    'train_epoch':10000,
    'learning_rate':0.02,
    'max_pool_size':4
})

params_rnn = tf.contrib.training.HParams(**params_rnn)

