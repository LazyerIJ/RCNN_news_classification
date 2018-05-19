import tensorflow as tf

params_share = ({
    'vocabulary_size' : 8000,
    'embedding_size' : 20,
    'max_len_sen':20,#35문장 수 자르기 --> 패딩 필요
    'max_len_word':50,#50이상 문장 길이 자르기 / 패딩
})
params_share = tf.contrib.training.HParams(**params_share)

cbow_params = ({
    'batch_size' : 500,
    'generations' : 5000,
    'model_learning_rate' : 0.001,
    'window_size' : 6,       # How many words to consider left and right.
    'save_embeddings_every' : 500,
    'print_valid_every' : 500,
    'print_loss_every' : 100,
    'num_sampled':400
})

params_cbow = tf.contrib.training.HParams(**cbow_params)

params_rnn = ({
    'split_rate':0.8,
    'batch_size':20,
    'num_filters':3,
    'maxpool_width':4,
    'bank_size':[2,3],
    'rnn_size':16,
    'layer_depth':3,
    'train_epoch':10000,
    'learning_rate':0.00001
})

params_rnn = tf.contrib.training.HParams(**params_rnn)

