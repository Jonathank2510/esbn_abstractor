import tensorflow as tf
from keras import layers
from Models.modules import Encoder_conv
from Models.ESBNLayer import ESBNLayer
from Models.TransformerModules import *

# This model reimplements the architecture by Webb using a custom made ESBN layer
class Dist3ESBN(tf.keras.Model):
    def __init__(self, z_size, key_size, hidden_size, y_dim=4):
        super().__init__()
        # Encoder
        self.encoder = Encoder_conv()

        # ESBN parameters
        self.z_size = z_size
        self.key_size = key_size
        self.hidden_size = hidden_size

        self.esbn_layer = ESBNLayer(self.key_size, self.hidden_size)
        self.out = layers.Dense(y_dim, activation="softmax")

        # Context normalization parameters
        self.gamma = tf.Variable(tf.ones(self.z_size))
        self.beta = tf.Variable(tf.zeros(self.z_size))


    def call(self, x_seq):
        #encode all images in sequence
        z_seq = tf.TensorArray(tf.float32, x_seq.shape[1])
        for t in tf.range(x_seq.shape[1]):
            x_t = tf.expand_dims(x_seq[:,t,:,:], 1)
            z_t = self.encoder(x_t)
            z_seq = z_seq.write(t, z_t)
        z_seq = tf.transpose(z_seq.stack(), (1, 0, 2))
        
        # Apply temporal context normalization
        z_seq = self.apply_context_norm(z_seq)
        # Append zero-element to input sequence to obtain predition from ESBN
        z_seq = tf.concat([z_seq, tf.zeros((tf.shape(z_seq)[0], 1, tf.shape(z_seq)[2]))], 1)
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
    

class Dist3Transformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder_conv()
        self.z_size = 128
        self.n_heads = 8
        self.dim_ff = 512
        self.n_layers = 1
        self.enc_layer = EncoderLayer(
                            d_model=self.z_size,
                            num_heads=self.n_heads,
                            dff=self.dim_ff)

        self.out_hidden = layers.Dense(256, activation="relu")
        self.y_out = layers.Dense(4, activation="softmax")
        self.positional_encoding = positional_encoding(length=5000, depth=self.z_size)

        # Context normalization parameters
        self.gamma = tf.Variable(tf.ones(self.z_size))
        self.beta = tf.Variable(tf.zeros(self.z_size))

    def call(self, x_seq):
        #encode all images in sequence
        z_seq = tf.TensorArray(tf.float32, x_seq.shape[1])
        for t in tf.range(x_seq.shape[1]):
            x_t = tf.expand_dims(x_seq[:,t,:,:], 1)
            z_t = self.encoder(x_t)
            z_seq = z_seq.write(t, z_t)
        z_seq = tf.transpose(z_seq.stack(), (1, 0, 2)) # (batch, seq, depth)

        # Apply temporal context normalization
        z_seq = self.apply_context_norm(z_seq)  

        # Run through stacked encoder
        length = z_seq.shape[1]
        #pos_encoding = positional_encoding(length=length, depth=self.z_size)

        #  This factor sets the relative scale of the embedding and positonal_encoding.
        z_seq *= tf.math.sqrt(tf.cast(self.z_size, tf.float32))
        z_seq += self.positional_encoding[tf.newaxis, :length, :]

        z_seq = self.enc_layer(z_seq)

        # Average over transformed sequences
        z_seq = tf.reduce_mean(z_seq, axis=1)

        out_hidden = self.out_hidden(z_seq)
        #tf.print("out_hidden:")
        #tf.print(out_hidden)
        out = self.y_out(out_hidden)
        return out
    
    def apply_context_norm(self, z_seq):
        eps = 1e-8
        z_mu = tf.reduce_mean(z_seq, 1, keepdims=True)
        z_sigma = tf.math.sqrt(tf.math.reduce_variance(z_seq, 1, keepdims=True) + eps)
        z_seq = (z_seq - z_mu) / z_sigma
        z_seq = (z_seq * self.gamma) + self.beta
        return z_seq