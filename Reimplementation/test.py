from ESBN_reimplementation import ESBN
import tensorflow as tf

model = ESBN(9)

model.call(tf.ones((5, 4, 1, 36, 36)))