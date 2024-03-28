import tensorflow as tf 
import datetime
import tqdm

# Define our metrics for performance logging
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

# Perform one epoch of training
def train_step(model, x_seq, targets, optimizer, loss_fn):
	with tf.GradientTape() as tape:
		y_pred_linear = model(x_seq)
		loss = loss_fn(targets, y_pred_linear)
	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
		
	train_loss(loss)
	train_accuracy(targets, y_pred_linear)
		

def test_step(model, all_imgs, test_loader, loss_fn):
	for seq_ind, y in test_loader:
		x_seq = tf.gather(all_imgs, seq_ind)
		y_pred_linear = model(x_seq)
		loss = loss_fn(y, y_pred_linear)
		test_loss(loss)
		test_accuracy(y, y_pred_linear)
		

def train(model, train_loader, test_loader, all_imgs, optimizer, loss_fn, epochs=50):
	# Define folders for logging
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
	test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	test_summary_writer = tf.summary.create_file_writer(test_log_dir)	

    # Train
	for epoch in range(epochs):
		for seq_ind, targets in tqdm.tqdm(train_loader, desc=f"Epoch: {epoch}"):
			x_seq = tf.gather(all_imgs, seq_ind)
			train_step(model, x_seq, targets, optimizer, loss_fn)

		with train_summary_writer.as_default():
			tf.summary.scalar("loss", train_loss.result(), step=epoch)
			tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)
		# Test
		test_step(model, all_imgs, test_loader, loss_fn)
		with test_summary_writer.as_default():
			tf.summary.scalar("loss", test_loss.result(), step=epoch)
			tf.summary.scalar("accuracy", test_accuracy.result(), step=epoch)