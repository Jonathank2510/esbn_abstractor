import os
import create_task
import tensorflow as tf 
from PIL import Image
import numpy as np 
import datetime

# Method for creating directory if it doesn't exist yet
def check_path(path):
	if not os.path.exists(path):
		os.mkdir(path)
		
# Set up loss and optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define our metrics for performance logging
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

# Perform one epoch of training
def train_step(model, all_imgs, train_loader):
	# Iterate over batches
	for batch_idx, (seq_ind, y) in enumerate(train_loader):
		x_seq = all_imgs[seq_ind,:,:]
		with tf.GradientTape() as tape:
			y_pred_linear, y_pred = model(x_seq)
			loss_fn = tf.keras.losses.CategoricalCrossentropy()
			loss = loss_fn(y_pred, y)
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
		
		train_loss(loss)
		train_accuracy(y, y_predictions)
		
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

def test_step(model, all_imgs, test_loader):
	for batch_ids, (seq_ind, y) in enumerate(test_loader):
		x_seq = all_imgs[seq_ind,:,:]
		y_pred_linear, y_pred = model(x_seq)
		loss = loss_fn(y, y_pred)
		test_loss(loss)
		test_accuracy(y, y_pred)
		

def train(model, train_loader, test_loader, all_imgs, n_shapes, epochs):
    # Train
	for epoch in range(epochs):
		# Training loop
		train_step(model, all_imgs, train_loader)
		with train_summary_writer.as_default():
			tf.summary.scalar("loss", test_loss.result(), step=epoch)
			tf.summary.scalar("accuracy", train_accuracy.result, step=epoch)
        
		
# Set train parameters
n_shapes = 50
epochs = 5

# Randomly assigns objects to training or test set
all_shapes = np.arange(n_shapes)

# Generate training and test sets
train_set, test_set = create_task()
# Load images
all_imgs = []
for i in range(n_shapes):
    img_fname = "./imgs/" + str(i) + ".png"
    img = tf.Tensor(np.array(Image.open(img_fname))) / 255.
    all_imgs.append(img)
all_imgs = tf.stack(all_imgs, 0)