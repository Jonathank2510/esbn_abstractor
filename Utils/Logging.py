import datetime
import tensorflow as tf

# Wrapper for more detailed logging
class BatchLoggingModel(tf.keras.Model):
    def __init__(self, model, log_dir):
        super().__init__()
        self.model = model
        self.train_writer = tf.summary.create_file_writer(log_dir + '/batch_level')

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = self.compiled_loss(y, y_pred)
            accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y, y_pred))
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        with self.train_writer.as_default(step=self._train_counter):
            tf.summary.scalar('batch_loss', loss)
            tf.summary.scalar('batch_accuracy', accuracy)
        return self.compute_metrics(x, y, y_pred, None)
    
    
    def call(self, x):
        x = self.model(x)
        return x