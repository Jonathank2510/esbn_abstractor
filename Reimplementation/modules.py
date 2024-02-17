import tensorflow as tf
from keras.layers import Conv2D, Dense, ReLu

class Encoder_conv(tf.keras.Model):
    def __init__(self, args):
        super(Encoder_conv, self).__init__()
        #Convolutional Layers
        self.conv1 = Conv2D(filters=32, kernel_size=4, strides=2, padding="same", input_shape=(32, 32, 1))
        self.conv2 = Conv2D(32, 4, 2, padding="same")
        self.conv2 = Conv2D(32, 4, 2, padding="same")
        #Dense Layers
        self.dense1 = Dense(4*4*32, 256)
        self.dense2 = Dense(256, 128)
        #Nonlinearities
        self.relu = ReLu()


    @tf.function
    def call(self, x):
        # Convolutional layers
        conv1_out = self.relu(self.conv1(x))
        conv2_out = self.relu(self.conv2(conv1_out))
        conv3_out = self.relu(self.conv3(conv2_out))
        # Flatten output of conv. net
        conv3_out_flat = tf.kears.Flatten()
        # Fully-connected layers
        dense1_out = self.relu(self.fc1(conv3_out_flat))
        dense2_out = self.relu(self.fc2(dense1_out))
        # Output
        z = dense2_out
        return z
    
#pre-trained word embedding
class Encoder_word(tf.keras.Model):
    pass