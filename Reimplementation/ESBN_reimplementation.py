import tensorflow as tf
from keras.layers import LSTM, Dense
from modules import *

#recreate distribution of three in tensorflow

class ESBN(tf.keras.Model):
    def __init__(self, task_gen, args):
        super(ESBN, self).__init__()
        #Encoder
        if args.encoder == "conv":
            self.encoder = Encoder_conv(args)
        elif args.encoder == "word":
            self.encoder = Encoder_word(args)

        #LSTM and output layers
        self.z_size = 128
        self.key_size = 256
        self.hidden_size = 512
        self.lstm = LSTM(self.hidden_size)
        self.key_w_out = Dense(self.key_size)
        self.g_out = Dense(1)
        self.confidence_gain = tf.Variable(1, trainable=True)
        self.confidence_bias = tf.Variable(0, trainable=True)
        self.y_out = Dense(task_gen.y_dim)

        