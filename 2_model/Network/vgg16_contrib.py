import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import *  

class vgg16:
    def __init__(self, imgs, keep_prob, weights=None):
        self.imgs = imgs
        if weights is not None:
            self.weights = np.load(weights)
        else:
            self.weights = None
        self.convlayers()
        self.fc_layers(keep_prob)
        self.probs = tf.nn.softmax(self.fc3l)

    def initialize_weight(self, name, stddev=0.01):
        if self.weights is not None:
            print(name, np.shape(self.weights[name]))
            return tf.constant_initializer(self.weights[name])
        else:
            if name[-1] == "W":
                return tf.random_normal_initializer(stddev=stddev)
            else:
                return tf.constant_initializer(1.0)

    def convlayers(self):
        with tf.contrib.slim.arg_scope([convolution2d], biases_initializer=tf.constant_initializer(0.0), 
            weights_regularizer=l2_regularizer(0.0005), trainable=True):

            # conv1_1
            self.conv1_1 = convolution2d(inputs=self.imgs, num_outputs=64, kernel_size=[3,3],
                weights_initializer=self.initialize_weight("conv1_1_W"),
                biases_initializer=self.initialize_weight("conv1_1_b"),
                stride=[1, 1], padding='SAME', scope='conv1_1')

            # conv1_2
            self.conv1_2 = convolution2d(inputs=self.conv1_1, num_outputs=64, kernel_size=[3,3], 
                weights_initializer=self.initialize_weight("conv1_2_W"),
                biases_initializer=self.initialize_weight("conv1_2_b"),
                    stride=[1, 1], padding='SAME', scope='conv1_2')

            # pool1
            self.pool1 = max_pool2d(inputs=self.conv1_2, kernel_size=2, stride=2, padding='SAME', scope='pool1')

            # conv2_1
            self.conv2_1 = convolution2d(inputs=self.pool1, num_outputs=128, kernel_size=[3,3], 
                weights_initializer=self.initialize_weight("conv2_1_W"),
                biases_initializer=self.initialize_weight("conv2_1_b"), 
                stride=[1, 1], padding='SAME', scope='conv2_1')

            # conv2_2
            self.conv2_2 = convolution2d(inputs=self.conv2_1, num_outputs=128, kernel_size=[3,3], 
                weights_initializer=self.initialize_weight("conv2_2_W"),
                biases_initializer=self.initialize_weight("conv2_2_b"), 
                stride=[1, 1], padding='SAME', scope='conv2_2')

            # pool2
            self.pool2 = max_pool2d(inputs=self.conv2_2, kernel_size=2, stride=2, padding='SAME', scope='pool2')

            # conv3_1
            self.conv3_1 = convolution2d(inputs=self.pool2, num_outputs=256, kernel_size=[3,3], 
                weights_initializer=self.initialize_weight("conv3_1_W"),
                biases_initializer=self.initialize_weight("conv3_1_b"), 
                stride=[1, 1], padding='SAME', scope='conv3_1')

            # conv3_2
            self.conv3_2 = convolution2d(inputs=self.conv3_1, num_outputs=256, kernel_size=[3,3], 
                weights_initializer=self.initialize_weight("conv3_2_W"),
                biases_initializer=self.initialize_weight("conv3_2_b"), 
                stride=[1, 1], padding='SAME', scope='conv3_2')

            # conv3_3
            self.conv3_3 = convolution2d(inputs=self.conv3_2, num_outputs=256, kernel_size=[3,3], 
                weights_initializer=self.initialize_weight("conv3_3_W"),
                biases_initializer=self.initialize_weight("conv3_3_b"), 
                stride=[1, 1], padding='SAME', scope='conv3_3')

            # pool3
            self.pool3 = max_pool2d(inputs=self.conv3_3, kernel_size=2, stride=2, padding='SAME', scope='pool3')

            # conv4_1
            self.conv4_1 = convolution2d(inputs=self.pool3, num_outputs=512, kernel_size=[3,3], 
                weights_initializer=self.initialize_weight("conv4_1_W"),
                biases_initializer=self.initialize_weight("conv4_1_b"), 
                stride=[1, 1], padding='SAME', scope='conv4_1')
                
            # conv4_2
            self.conv4_2 = convolution2d(inputs=self.conv4_1, num_outputs=512, kernel_size=[3,3], 
                weights_initializer=self.initialize_weight("conv4_2_W"),
                biases_initializer=self.initialize_weight("conv4_2_b"), 
                stride=[1, 1], padding='SAME', scope='conv4_2')
                
            # conv4_3
            self.conv4_3 = convolution2d(inputs=self.conv4_2, num_outputs=512, kernel_size=[3,3], 
                weights_initializer=self.initialize_weight("conv4_3_W"),
                biases_initializer=self.initialize_weight("conv4_3_b"), 
                stride=[1, 1], padding='SAME', scope='conv4_3')

            # pool4
            self.pool4 = max_pool2d(inputs=self.conv4_3, kernel_size=2, stride=2, padding='SAME', scope='pool4')

            # conv5_1
            self.conv5_1 = convolution2d(inputs=self.pool4, num_outputs=512, kernel_size=[3,3], 
                weights_initializer=self.initialize_weight("conv5_1_W"),
                biases_initializer=self.initialize_weight("conv5_1_b"), 
                stride=[1, 1], padding='SAME', scope='conv5_1')

            self.conv5_2 = convolution2d(inputs=self.conv5_1, num_outputs=512, kernel_size=[3,3], 
                weights_initializer=self.initialize_weight("conv5_2_W"),
                biases_initializer=self.initialize_weight("conv5_2_b"), 
                stride=[1, 1], padding='SAME', scope='conv5_2')
                
            self.conv5_3 = convolution2d(inputs=self.conv5_2, num_outputs=512, kernel_size=[3,3], 
                weights_initializer=self.initialize_weight("conv5_3_W"),
                biases_initializer=self.initialize_weight("conv5_3_b"), 
                stride=[1, 1], padding='SAME', scope='conv5_3')

            # pool5
            self.pool5 = max_pool2d(inputs=self.conv5_3, kernel_size=2, stride=2, padding='SAME', scope='pool5')

    def fc_layers(self, keep_prob=1.0):
        shape = int(np.prod(self.pool5.get_shape()[1:]))
        flat = tf.reshape(self.pool5, [-1, shape])
        with tf.contrib.slim.arg_scope([fully_connected], weights_regularizer=l2_regularizer(0.0005), trainable=True):

            # fc1
            self.fc1 = fully_connected(flat, 4096, scope='fc1', 
                weights_initializer=self.initialize_weight("fc6_W"), 
                biases_initializer=self.initialize_weight("fc6_b"))

            self.drop1 = tf.nn.dropout(self.fc1, keep_prob, name='drop1')

            # fc2
            self.fc2 = fully_connected(self.drop1, 4096, scope='fc2',
                weights_initializer=self.initialize_weight("fc7_W"), 
                biases_initializer=self.initialize_weight("fc7_b"))

            self.drop2 = tf.nn.dropout(self.fc2, keep_prob, name='drop2')

            # fc3
            with tf.name_scope('fc3') as scope:
                fc3w = tf.Variable(tf.random_normal([4096, 8],
                                                             dtype=tf.float32,
                                                             stddev=0.01), name='fc3_W')

                fc3b = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float32),
                                     trainable=True, name='fc3_b')
                self.fc3l = tf.nn.bias_add(tf.matmul(self.drop2, fc3w), fc3b)