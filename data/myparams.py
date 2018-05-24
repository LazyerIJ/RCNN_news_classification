import tensorflow as tf

params_share = ({
    'vocabulary_size' : 1000,
    'embedding_size' : 10,
    'max_len_sen':20,#35문장 수 자르기 --> 패딩 필요
    'max_len_word':30,#50이상 문장 길이 자르기 / 패딩
    'label_num':5
})
params_share = tf.contrib.training.HParams(**params_share)

cbow_params = ({
    'batch_size' : 500,
    'generations' : 3000,
    'model_learning_rate' : 0.001,
    'window_size' : 4,       # How many words to consider left and right.
    'save_embeddings_every' : 500,
    'print_valid_every' : 500,
    'print_loss_every' : 100,
    'num_sampled':400
})

params_cbow = tf.contrib.training.HParams(**cbow_params)

params_rnn = ({
    'split_rate':0.8,
    'batch_size':20,
    'num_filters':60,
    'maxpool_width':4,
    'bank_size':[30],
    'rnn_size':64,
    'layer_depth':2,
    'train_epoch':10000,
    'learning_rate':0.001
})

params_rnn = tf.contrib.training.HParams(**params_rnn)

