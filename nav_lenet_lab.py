# LeNet lab solution from Udacity Self Driving Card ND
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def load_mnist_and_pad_with_zeros():
	mnist = input_data.read_data_sets('MNIST_data', reshape=False)
	X_train, y_train = mnist.train.images, mnist.train.labels
	X_validation, y_validation = mnist.validation.images, mnist.validation.labels
	X_test, y_test = mnist.test.images, mnist.test.labels

	print()
	print('Image Shape: {}'.format(X_train[0].shape))
	print('Training Set: {} samples'.format(len(X_train)))
	print('Validation Set: {} samples'.format(len(X_validation)))
	print('Test Set: {} samples'.format(len(X_test)))
	print()

	# MNIST data that TF preloads comes with 28x28x1 images.
	# However, LeNet architecture accepts only 32x32xC images where C is the number of color channels
	# So we will use np.pad to reshape the data (we append 2 rows of zeros top and bottom and 2 columns of zeros left and right)
	# Pad images with zeros	
	X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
	X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
	X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
	print('Updated Image Shape: {}'.format(X_train[0].shape))
	print()
	return X_train, y_train, X_validation, y_validation, X_test, y_test


##############################
# Load the MNIST data
##############################
X_train, y_train, X_validation, y_validation, X_test, y_test = load_mnist_and_pad_with_zeros()


##############################
# Setup TensorFlow
##############################
EPOCHS = 10
BATCH_SIZE = 128

##############################
# Helper functions
##############################
def get_weights(shape, mu=0.0, sigma=0.01):
	w = tf.Variable(tf.truncated_normal(shape=shape, mean=mu, stddev=sigma))
	return w

def get_bias(n_labels):
	return tf.Variable(tf.zeros(n_labels))

##############################
# Implement LeNet-5
##############################
def LeNet(x):
	# Arguments used for tf.truncated_normal -- which is used to randomly initialize weights and biases for each layer
	mu = 0.0
	sigma = 0.1

	# Layer1: Convolutional: Input = 32x32x1, Output = 28x28x6
	l1_w = get_weights(shape=(5,5,1,6), mu=mu, sigma=sigma)
	l1_b = get_bias(6)
	l1_conv = tf.nn.conv2d(x, l1_w, strides=[1,1,1,1], padding='VALID') + l1_b

	# Layer1: Activation
	l1_conv = tf.nn.relu(l1_conv)

	# Layer1: Pooling: Input = 28x28x6, Output = 14x14x6
	l1_conv = tf.nn.max_pool(l1_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

	# Layer2: Convolutional: Input = 14x14x6, Output = 10x10x16
	l2_w = get_weights(shape=(5,5,6,16), mu=mu, sigma=sigma)
	l2_b = get_bias(16)
	l2_conv = tf.nn.conv2d(l1_conv, l2_w, strides=[1,1,1,1], padding='VALID') + l2_b

	# Layer2: Activation
	l2_conv = tf.nn.relu(l2_conv)

	# Layer2: Pooling: Input = 10x10x16, Output = 5x5x16
	l2_conv = tf.nn.max_pool(l2_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

	# Flatten the output from layer2 to create a 1D input as the next layer is fully connected layer
	flattened_l2_conv = tf.contrib.layers.flatten(l2_conv)

	# Layer3: Fully Connected: Input = 400, Output = 120
	l3_w = get_weights(shape=(400,120), mu=mu, sigma=sigma)
	l3_b = get_bias(120)
	l3_fc = tf.matmul(flattened_l2_conv, l3_w) + l3_b

	# Layer3: Activation
	l3_fc = tf.nn.relu(l3_fc)

	# Layer4: Fully Connected: Input = 120, Output = 84
	l4_w = get_weights(shape=(120,84), mu=mu, sigma=sigma)
	l4_b = get_bias(84)
	l4_fc = tf.matmul(l3_fc, l4_w) + l4_b

	# Layer4: Activation
	l4_fc = tf.nn.relu(l4_fc)

	# Layer5: Fully Connected (Logits): Input = 84, Output = 10
	l5_w = get_weights(shape=(84,10), mu=mu, sigma=sigma)
	l5_b = get_bias(10)
	l5_fc_logits = tf.matmul(l4_fc, l5_w) + l5_b

	return l5_fc_logits


##############################
# Features and Labels
# x is a placeholder for a batch of input images. y is a placeholder for a batch of labels
##############################
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

##############################
# Training Pipeline
# Create a training pipeline that uses the model to classify MNIST data
##############################
learning_rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_op = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss_op)

##############################
# Model Evaluation
# Evaluate the loss and accuracy of the model for a given dataset
##############################
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
# mistakes = tf.not_equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
	sess = tf.get_default_session()
	accuracy_value = sess.run(fetches=accuracy_op, feed_dict={x: X_data, y: y_data})
	return accuracy_value

##############################
# Train the Model
# Run the training data through the training pipeline
# Before each epoch, shuffle the training data
# After each epoch, measure the loss and accuracy on validation set
# Save the model after training
##############################
with tf.Session() as sess:
	init_op = tf.initialize_all_variables()
	sess.run(init_op)

	num_examples = len(X_train)

	print("Training ...")
	print()
	for j in range(EPOCHS):
		X_train, y_train = shuffle(X_train, y_train)
		for batch_start_offset in range(0, num_examples, BATCH_SIZE):
			batch_end_offset = batch_start_offset + BATCH_SIZE
			current_batch_x = X_train[batch_start_offset:batch_end_offset]
			current_batch_y = y_train[batch_start_offset:batch_end_offset]
			sess.run(fetches=training_op, feed_dict={x: current_batch_x, y: current_batch_y})

		validation_accuracy = evaluate(X_validation, y_validation)
		print('Epoch {} complete and validation accuracy = {:.3f}'.format(j, validation_accuracy))
		print()

	saver.save(sess, './lenet')
	print('Saved the model')
