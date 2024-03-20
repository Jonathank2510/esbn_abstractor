import tensorflow as tf

# ESBN layer generates abstract keys and returns last key for sequence
class ESBN(tf.keras.layers.Layer):
    def __init__(self, key_size, hidden_size):
        super(ESBN, self).__init__()
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.lstm_cell = tf.keras.layers.LSTMCell(hidden_size)
        self.key_w_out = tf.keras.layers.Dense(key_size)
        self.g_out = tf.keras.layers.Dense(1, activation="sigmoid")
        self.confidence_gain = tf.Variable(1, dtype=tf.float32)
        self.confidence_bias = tf.Variable(0, dtype=tf.float32)
    
    # States are of the form (hidden, cell_state)
    #@tf.function()
    def call(self, inputs):
        M_k = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        M_v = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        key_r = tf.zeros((inputs.shape[0], 1, self.key_size + 1))
        # Initial states for LSTM
        hidden = tf.zeros((inputs.shape[0], self.hidden_size))
        cell_state = tf.zeros((inputs.shape[0], self.hidden_size))

        # Loop over timesteps
        for t in tf.range(len(inputs[1])):
            # Get value at timestep t
            x = tf.expand_dims(inputs[:,t,:], 1)
            # Controller
            lstm_out, (hidden, cell_state) = self.lstm_cell(tf.squeeze(key_r, 1), [hidden, cell_state])
            # Generate write key
            key_w = self.key_w_out(lstm_out)
            # Gate
            g = tf.expand_dims(self.g_out(lstm_out), 2)
            if t > 0:
                # Compute dot product similarities with memory values
                M_v_tensor = tf.transpose(M_v.stack(), (1, 0, 2))
                dot_product = tf.reduce_sum((x * M_v_tensor), 2, keepdims=True)
                w_k = tf.math.softmax(dot_product)
                c_k = tf.math.sigmoid(dot_product * self.confidence_gain + self.confidence_bias)
                # Generate read key as a weighted sum of past keys
                M_k_tensor = tf.transpose(M_k.stack(), (1, 0, 2))
                weighted_sum = tf.reduce_sum(w_k + tf.concat([M_k_tensor, c_k], axis=2), axis=1, keepdims=True)
                key_r = g * weighted_sum
            # Write to memory
            M_k.write(t, key_w)
            M_v.write(t, tf.squeeze(x, 1))

        # Return last state of lstm
        return lstm_out