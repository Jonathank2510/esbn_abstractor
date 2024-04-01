import tensorflow as tf

# ESBN layer generates abstract keys and returns last key for sequence
class ESBNLayer(tf.keras.layers.Layer):
    def __init__(self, key_size, hidden_size, return_memory=False):
        super(ESBNLayer, self).__init__()
        self.return_memory = return_memory
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.lstm_cell = tf.keras.layers.LSTMCell(hidden_size)
        self.key_w_out = tf.keras.layers.Dense(key_size)
        self.g_out = tf.keras.layers.Dense(1, activation="sigmoid")
        self.confidence_gain = tf.Variable(1, dtype=tf.float32)
        self.confidence_bias = tf.Variable(0, dtype=tf.float32)
    
    def call(self, inputs):
        M_k = tf.TensorArray(tf.float32, size=0, 
                             element_shape=tf.TensorShape((32, self.key_size)),
                             dynamic_size=True)
        M_v = tf.TensorArray(tf.float32, size=0, 
                             element_shape=tf.TensorShape((32, inputs.shape[2])),
                             dynamic_size=True)
        
        key_r = tf.zeros((tf.shape(inputs)[0], 1, self.key_size + 1))
        # Initial states for LSTM
        hidden = tf.zeros((tf.shape(inputs)[0], self.hidden_size))
        cell_state = tf.zeros((tf.shape(inputs)[0], self.hidden_size))
        #  Pre define loop variables for graph executions
        lstm_out = tf.zeros((tf.shape(inputs)[0], self.hidden_size))


        # Loop over timesteps
        for t in tf.range(inputs.shape[1]):
            # Get value at timestep t
            x = tf.expand_dims(inputs[:,t,:], 1)
            # Controller
            lstm_out, (hidden, cell_state) = self.lstm_cell(tf.squeeze(key_r, 1), states=[hidden, cell_state])
            #tf.print("lstm_out")
            #tf.print(lstm_out[0])
            # Generate write key
            key_w = self.key_w_out(lstm_out)
            #tf.print("key_w:")
            #tf.print(key_w[0])
            # Gate
            g = tf.expand_dims(self.g_out(lstm_out), 2)
            #tf.print("g:")
            #tf.print(g[0])
            if t == 0:
                key_r = tf.zeros((tf.shape(inputs)[0], 1, self.key_size + 1))
                
            else:
                # Compute dot product similarities with memory values
                similarities = tf.reduce_sum((x * tf.transpose(M_v.stack(), (1, 0, 2))), 2, keepdims=True)
                w_k = tf.keras.activations.softmax(similarities, axis=1)
                c_k = tf.keras.activations.sigmoid(similarities * self.confidence_gain + self.confidence_bias)
                # Generate read key as a weighted sum of past keys
                weighted_sum = tf.reduce_sum(w_k * tf.concat([tf.transpose(M_k.stack(), (1, 0, 2)), c_k], axis=2), axis=1, keepdims=True)
                key_r = g * weighted_sum
        
            # Write to memory
            M_k = M_k.write(t, key_w)
            M_v = M_v.write(t, tf.squeeze(x, 1))
            
            
        # Return last state of lstm and external memory (optionally)
        if(self.return_memory):
            pass
            #return lstm_out, (tf.transpose(M_k.stack(), (1, 0, 2)), tf.transpose(M_v.stack(), (1, 0, 2)))
        else:
            return lstm_out