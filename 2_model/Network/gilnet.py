import tensorflow as tf
import numpy as np

class gilnet:
    def __init__(self, imgs, keep_prob):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers(keep_prob)
        self.probs = tf.nn.softmax(self.fc3l)

    def convlayers(self):
        self.parameters = []

        # conv1
        with tf.name_scope('conv1') as scope:
            kernel = tf.get_variable(name='conv1_W', shape=[7, 7, 3, 96], 
                initializer=tf.random_normal_initializer(stddev=0.01),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))

            conv = tf.nn.conv2d(self.imgs, kernel, [1, 4, 4, 1], padding='VALID')
            biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                                 trainable=True, name='conv1_b')
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool1')

        self.norm1 = tf.nn.local_response_normalization(self.pool1, 5, alpha=0.0001, beta=0.75, name='norm1')

        # conv2
        with tf.name_scope('conv2') as scope:
            kernel = tf.get_variable(name='conv2_W', shape=[5, 5, 96, 256], 
                initializer=tf.random_normal_initializer(stddev=0.01),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))

            conv = tf.nn.conv2d(self.norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='conv2_b')
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool2')

        self.norm2 = tf.nn.local_response_normalization(self.pool2, 5, alpha=0.0001, beta=0.75, name='norm2')

        # conv3
        with tf.name_scope('conv3') as scope:
            kernel = tf.get_variable(name='conv3_W', shape=[3, 3, 256, 384], 
                initializer=tf.random_normal_initializer(stddev=0.01),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))

            conv = tf.nn.conv2d(self.norm2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                                 trainable=True, name='conv3_b')
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='pool3')

    def fc_layers(self, keep_prob=1.0):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool3.get_shape()[1:]))
            fc1w = tf.get_variable(name='fc1_W', shape=[shape, 512], 
                initializer=tf.random_normal_initializer(stddev=0.005),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))

            fc1b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='fc1_b')
            pool3_flat = tf.reshape(self.pool3, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        self.drop1 = tf.nn.dropout(self.fc1, keep_prob, name='drop1') 

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.get_variable(name='fc2_W', shape=[512, 512], 
                initializer=tf.random_normal_initializer(stddev=0.005),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))

            fc2b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='fc2_b')
            fc2l = tf.nn.bias_add(tf.matmul(self.drop1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        self.drop2 = tf.nn.dropout(self.fc2, keep_prob, name='drop2') 

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.random_normal([512, 8],
                                                         dtype=tf.float32,
                                                         stddev=0.01), name='fc3_W')

            fc3b = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float32),
                                 trainable=True, name='fc3_b')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.drop2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]