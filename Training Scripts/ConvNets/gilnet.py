import tensorflow as tf
import numpy as np

# weights initializers
#conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()
#linear_initializer = tf.contrib.layers.xavier_initializer()
conv_initializer = tf.random_normal_initializer(stddev=0.01)
linear_initializer = tf.random_normal_initializer(stddev=0.01)

class gilnet():
	def __init__(self, bgr_mean, num_classes=8, weight_decay=5e-4, enable_moving_average=False):
		# input tensors
		self.input_x = tf.placeholder(tf.float32, [None, 227, 227, 3], name="input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		
		# parameters
		self.parameters = []

		# zero-mean input
		with tf.name_scope('zero_mean') as scope:
			mean = tf.constant(bgr_mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
			images = self.input_x-mean
		
		# conv1
		with tf.variable_scope("conv1") as scope:
			kernel = tf.get_variable(name='W', shape=[7, 7, 3, 96], 
				initializer=conv_initializer,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))

			conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='VALID')
			
			biases = tf.get_variable('b', [96], initializer=tf.constant_initializer(0.0))
			out = tf.nn.bias_add(conv, biases)
			self.conv1 = tf.nn.relu(out, name='conv')
			self.parameters += [kernel, biases]
			
		# pool1
		self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
									padding='VALID', name='pool1')
							   
		self.norm1 = tf.nn.local_response_normalization(self.pool1, 5, alpha=0.0001, beta=0.75, name='norm1')

		# conv2
		with tf.variable_scope("conv2") as scope:
			kernel = tf.get_variable(name='W', shape=[5, 5, 96, 256], 
				initializer=conv_initializer,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
			conv = tf.nn.conv2d(self.norm1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.get_variable('b', [256], initializer=tf.constant_initializer(0.0))
			out = tf.nn.bias_add(conv, biases)
			self.conv2 = tf.nn.relu(out, name='conv')
			self.parameters += [kernel, biases]
		
		# pool2
		self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
									padding='VALID', name='pool2')

		self.norm2 = tf.nn.local_response_normalization(self.pool2, 5, alpha=0.0001, beta=0.75, name='norm2')
		
		# conv3
		with tf.variable_scope("conv3") as scope:
			kernel = tf.get_variable(name='W', shape=[3, 3, 256, 384], 
				initializer=conv_initializer,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
			conv = tf.nn.conv2d(self.norm2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.get_variable('b', [384], initializer=tf.constant_initializer(0.0))
			out = tf.nn.bias_add(conv, biases)
			self.conv3 = tf.nn.relu(out, name='conv')
			self.parameters += [kernel, biases]

		# pool3
		self.pool3 = tf.nn.max_pool(self.conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
									padding='VALID', name='pool3')
									
		shape = int(np.prod(self.pool3.get_shape()[1:]))
		flat = tf.reshape(self.pool3, [-1, shape])
									
		# fc1
		with tf.variable_scope('fc1'):
			w = tf.get_variable('W', [shape, 512], initializer=linear_initializer,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
			b = tf.get_variable('b', [512], initializer=tf.constant_initializer(0.0))
			out = tf.matmul(flat, w) + b
			self.fc1 = tf.nn.relu(out)
			self.drop1 = tf.nn.dropout(self.fc1, self.dropout_keep_prob, name='drop1')
			self.parameters += [w, b]

		# fc2
		with tf.variable_scope('fc2'):
			w = tf.get_variable('W', [self.drop1.get_shape()[1], 512], initializer=linear_initializer,
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
			b = tf.get_variable('b', [512], initializer=tf.constant_initializer(0.0))
			out = tf.matmul(self.drop1, w) + b
			self.fc2 = tf.nn.relu(out)
			self.drop2 = tf.nn.dropout(self.fc2, self.dropout_keep_prob, name='drop2')
			self.parameters += [w, b]

		# fc3
		with tf.variable_scope('fc3'):
			w = tf.get_variable('W', [self.drop2.get_shape()[1], num_classes], initializer=linear_initializer)
			b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
			self.fc3 = tf.matmul(self.drop2, w) + b
			self.parameters += [w, b]
		
		# Calculate Mean cross-entropy loss
		with tf.name_scope("loss"):
			self.predictions = tf.argmax(self.fc3, 1, name="predictions")
			losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.fc3, labels = self.input_y)
			regularization_losses = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			if enable_moving_average:
				total_loss = tf.reduce_mean(losses) + regularization_losses
				moving_averages = tf.train.ExponentialMovingAverage(0.9)
				moving_averages_op = moving_averages.apply([tf.reduce_mean(losses)] + [total_loss])
				with tf.control_dependencies([moving_averages_op]):
					self.loss = tf.identity(total_loss)
			else:
				self.loss = tf.reduce_mean(losses) + regularization_losses

		# Accuracy
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")