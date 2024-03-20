import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, rnn_type, m_hidden, x_hidden, ):
        super(Encoder, self).__init__()
        self.rnn_type = rnn_type

    def __call__(self, inputs):
        pass