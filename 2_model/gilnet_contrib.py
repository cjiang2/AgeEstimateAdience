import tensorflow as tf
from tensorflow.contrib.layers import * 
import numpy as np

class gilnet:
    def __init__(self, imgs, keep_prob):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers(keep_prob)
        self.probs = tf.nn.softmax(self.fc3l)

    def convlayers(self):
        with tf.contrib.slim.arg_scope([convolution2d], biases_initializer=tf.constant_initializer(0.0), 
            weights_regularizer=l2_regularizer(0.0005),
            weights_initializer=tf.random_normal_initializer(stddev=0.01), trainable=True):

            # conv1
            self.conv1 = convolution2d(inputs=self.imgs, num_outputs=96, kernel_size=[7,7], 
                    stride=[4, 4], padding='VALID', scope='conv1')

            self.pool1 = max_pool2d(inputs=self.conv1, kernel_size=3, stride=2, padding='VALID', scope='pool1')

            self.norm1 = tf.nn.local_response_normalization(self.pool1, 5, alpha=0.0001, beta=0.75, name='norm1')

            # conv2
            self.conv2 = convolution2d(inputs=self.norm1, num_outputs=256, kernel_size=[5, 5], 
                stride=[1, 1], padding='SAME', scope='conv2')

            self.pool2 = max_pool2d(inputs=self.conv2, kernel_size=3, stride=2, padding='VALID', scope='pool2')

            self.norm2 = tf.nn.local_response_normalization(self.pool2, 5, alpha=0.0001, beta=0.75, name='norm2')

            # conv3
            self.conv3 = convolution2d(self.norm2, num_outputs=384, kernel_size=[3, 3], 
                stride=[1, 1], padding='SAME', scope='conv3')  
              
            self.pool3 = max_pool2d(self.conv3, kernel_size=3, stride=2, padding='VALID', scope='pool3')  

    def fc_layers(self, keep_prob=1.0):
        shape = int(np.prod(self.pool3.get_shape()[1:]))
        flat = tf.reshape(self.pool3, [-1, shape])
        with tf.contrib.slim.arg_scope([fully_connected], biases_initializer=tf.constant_initializer(1.0), 
            weights_regularizer=l2_regularizer(0.0005),
            weights_initializer=tf.random_normal_initializer(stddev=0.005), trainable=True):

            # fc1
            self.fc1 = fully_connected(flat, 512, scope='fc1')
            self.drop1 = tf.nn.dropout(self.fc1, keep_prob, name='drop1')

            # fc2
            self.fc2 = fully_connected(self.drop1, 512, scope='fc2')
            self.drop2 = tf.nn.dropout(self.fc2, keep_prob, name='drop2')

            # fc3
            with tf.name_scope('fc3') as scope:
                fc3w = tf.Variable(tf.random_normal([512, 8],
                                                             dtype=tf.float32,
                                                             stddev=0.01), name='fc3_W')

                fc3b = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float32),
                                     trainable=True, name='fc3_b')
                self.fc3l = tf.nn.bias_add(tf.matmul(self.drop2, fc3w), fc3b)