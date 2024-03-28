import tensorflow as tf
from keras.layers import LSTM, Dense
from Models.modules import Encoder_conv
from Models.ESBNLayer import ESBNLayer

#recreate distribution of three in tensorflow
# TODO: implement temporal context norm, use @tf.function decorator
class ESBN(tf.keras.Model):
    def __init__(self, y_dim):
        super(ESBN, self).__init__()
        # Encoder
        self.encoder = Encoder_conv()
        

        # LSTM and output layers
        self.z_size = 128
        self.key_size = 256
        self.hidden_size = 512
        self.lstm = LSTM(self.hidden_size, return_state=True)
        self.key_w_out = Dense(self.key_size, activation="relu")
        self.g_out = Dense(1, activation="sigmoid")
        self.confidence_gain = tf.Variable(1, dtype=tf.float32)
        self.confidence_bias = tf.Variable(0, dtype=tf.float32)
        self.y_out = Dense(y_dim, activation="softmax")

        # Context normalization parameters
        self.gamma = tf.Variable(tf.ones(self.z_size), name="tcn_gamma")
        self.beta = tf.Variable(tf.zeros(self.z_size), name="tcn_beta")


    #@tf.function
    def call(self, x_seq):
        #encode all images in sequence
        z_seq = []
        for t in range(x_seq.shape[1]):
            x_t = tf.expand_dims(x_seq[:,t,:,:], 1)
            z_t = self.encoder(x_t)
            z_seq.append(z_t)
        # Apply temporal context normalization
        z_seq = self.apply_context_norm(z_seq)
        #stack embeddings along the first dimension (timesteps)
        z_seq = tf.stack(z_seq, 1)
        # Initialize retrieved key vector
        key_r = tf.zeros((x_seq.shape[0], 1, self.key_size + 1))
        self.M_k = []
        self.M_v = []
        # Initialize lstm
        hidden = tf.zeros((x_seq.shape[0], self.hidden_size))
        cell_state = tf.zeros((x_seq.shape[0], self.hidden_size))
        #iterate over timesteps
        for t in range(x_seq.shape[1] + 1):
            # Image embedding
            if t == x_seq.shape[1]:
                z_t = tf.zeros((x_seq.shape[0], 1, self.z_size))
            else:
                z_t = tf.expand_dims(z_seq[:,t,:], 1)
            # Controller
            # LSTM
            lstm_out, hidden, cell_state = self.lstm(key_r, initial_state=(cell_state, hidden))
            # Key output layers
            key_w = tf.expand_dims(self.key_w_out(lstm_out), 1)
            # Gates
            g = tf.expand_dims(self.g_out(lstm_out), 2)
            # Task output layer
            y_pred_linear = self.y_out(lstm_out)
            y_pred = tf.argmax(y_pred_linear, 1)
            # Read from memory
            if t == 0:
                key_r = tf.zeros((x_seq.shape[0], 1, self.key_size + 1))
            else:
                # Read key
                similarity = tf.reduce_sum((z_t * M_v), 2, keepdims=True)
                w_k = tf.keras.activations.softmax(similarity, axis=1)
                c_k = tf.keras.activations.sigmoid(tf.reduce_sum((z_t * M_v), 2, keepdims=True) * self.confidence_gain + self.confidence_bias)
                weighted_sum = tf.reduce_sum(w_k * tf.concat([M_k, c_k], axis=2), axis=1, keepdims=True)
                key_r = g * weighted_sum

            # Write to memory
            if t == 0:
                M_k = key_w
                M_v = z_t
            else:
                M_k = tf.concat([M_k, key_w], 1)
                M_v = tf.concat([M_v, z_t], 1)

            if t == x_seq.shape[1]:
                self.last_M_k = M_k
                self.last_M_v = M_v
            if t == x_seq.shape[1] - 1:
                self.second_last_sim = similarity
                self.secon_last_w = w_k

        return y_pred_linear
    
    def apply_context_norm(self, z_seq):
        eps = 1e-8
        z_mu = tf.reduce_mean(z_seq, 1, keepdims=True)
        z_sigma = tf.math.sqrt(tf.math.reduce_variance(z_seq, 1, keepdims=True) + eps)
        z_seq = (z_seq - z_mu) / z_sigma
        z_seq = (z_seq * self.gamma) + self.beta
        return z_seq





class Model2(tf.keras.Model):
    def __init__(self, y_dim):
        super(Model2, self).__init__()
        # Encoder
        self.encoder = Encoder_conv()

        # ESBN parameters
        self.z_size = 128
        self.key_size = 256
        self.hidden_size = 512

        self.esbn_layer = ESBNLayer(self.key_size, self.hidden_size)
        self.out = Dense(y_dim, activation="softmax")

        # Context normalization parameters
        self.gamma = tf.Variable(tf.ones(self.z_size))
        self.beta = tf.Variable(tf.zeros(self.z_size))


    def call(self, x_seq):
        # TODO: make code more tf-style
        #encode all images in sequence
        z_seq = []
        for t in range(x_seq.shape[1]):
            x_t = tf.expand_dims(x_seq[:,t,:,:], 1)
            z_t = self.encoder(x_t)
            z_seq.append(z_t)
        # Apply temporal context normalization
        z_seq = self.apply_context_norm(z_seq)
        #stack embeddings along the first dimension (timesteps)
        z_seq = tf.stack(z_seq, 1)
        # Append zero-element to input sequence to obtain predition from ESBN
        z_seq = tf.concat([z_seq, tf.zeros((z_seq.shape[0], 1, z_seq.shape[2]))], 1)
        # Get (purely relational processed) predictions from ESBN layer
        esbn_out = self.esbn_layer(z_seq)
        logits = self.out(esbn_out)
        return logits
        

    def apply_context_norm(self, z_seq):
            eps = 1e-8
            z_mu = tf.reduce_mean(z_seq, 1, keepdims=True)
            z_sigma = tf.math.sqrt(tf.math.reduce_variance(z_seq, 1, keepdims=True) + eps)
            z_seq = (z_seq - z_mu) / z_sigma
            z_seq = (z_seq * self.gamma) + self.beta
            return z_seq