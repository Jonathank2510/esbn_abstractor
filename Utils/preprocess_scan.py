import tensorflow as tf

max_vocab_size = 100

command_text_processor = tf.keras.layers.TextVectorization(
    standardize=None
)
command_text_processor.adapt()