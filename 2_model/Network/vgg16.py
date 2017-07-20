import tensorflow as tf
import numpy as np

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
        self.parameters = []

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.get_variable(name='conv1_1_W', shape=[3, 3, 3, 64], 
                initializer=self.initialize_weight('conv1_1_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))

            conv = tf.nn.conv2d(self.imgs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv1_1_b', shape=[64], 
                initializer=self.initialize_weight('conv1_1_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.get_variable(name='conv1_2_W', shape=[3, 3, 64, 64], 
                initializer=self.initialize_weight('conv1_2_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv1_2_b', shape=[64], 
                initializer=self.initialize_weight('conv1_2_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.get_variable(name='conv2_1_W', shape=[3, 3, 64, 128], 
                initializer=self.initialize_weight('conv2_1_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))         
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv2_1_b', shape=[128], 
                initializer=self.initialize_weight('conv2_1_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.get_variable(name='conv2_2_W', shape=[3, 3, 128, 128], 
                initializer=self.initialize_weight('conv2_2_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv2_2_b', shape=[128], 
                initializer=self.initialize_weight('conv2_2_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.get_variable(name='conv3_1_W', shape=[3, 3, 128, 256], 
                initializer=self.initialize_weight('conv3_1_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv3_1_b', shape=[256], 
                initializer=self.initialize_weight('conv3_1_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.get_variable(name='conv3_2_W', shape=[3, 3, 256, 256], 
                initializer=self.initialize_weight('conv3_2_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv3_2_b', shape=[256], 
                initializer=self.initialize_weight('conv3_2_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.get_variable(name='conv3_3_W', shape=[3, 3, 256, 256], 
                initializer=self.initialize_weight('conv3_3_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv3_3_b', shape=[256], 
                initializer=self.initialize_weight('conv3_3_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.get_variable(name='conv4_1_W', shape=[3, 3, 256, 512], 
                initializer=self.initialize_weight('conv4_1_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv4_1_b', shape=[512], 
                initializer=self.initialize_weight('conv4_1_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.get_variable(name='conv4_2_W', shape=[3, 3, 512, 512], 
                initializer=self.initialize_weight('conv4_2_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv4_2_b', shape=[512], 
                initializer=self.initialize_weight('conv4_2_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.get_variable(name='conv4_3_W', shape=[3, 3, 512, 512], 
                initializer=self.initialize_weight('conv4_3_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv4_3_b', shape=[512], 
                initializer=self.initialize_weight('conv4_3_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.get_variable(name='conv5_1_W', shape=[3, 3, 512, 512], 
                initializer=self.initialize_weight('conv5_1_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv5_1_b', shape=[512], 
                initializer=self.initialize_weight('conv5_1_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.get_variable(name='conv5_2_W', shape=[3, 3, 512, 512], 
                initializer=self.initialize_weight('conv5_2_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv5_2_b', shape=[512], 
                initializer=self.initialize_weight('conv5_2_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.get_variable(name='conv5_3_W', shape=[3, 3, 512, 512], 
                initializer=self.initialize_weight('conv5_3_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv5_3_b', shape=[512], 
                initializer=self.initialize_weight('conv5_3_b'))
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool5')

    def fc_layers(self, keep_prob=1.0):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.get_variable(name='fc1_W', shape=[shape, 4096], 
                initializer=self.initialize_weight('fc6_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))
            fc1b = tf.get_variable(name='fc1_b', shape=[4096], 
                initializer=self.initialize_weight('fc6_b'))
            pool3_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        self.drop1 = tf.nn.dropout(self.fc1, keep_prob, name='drop1') 

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.get_variable(name='fc2_W', shape=[4096, 4096], 
                initializer=self.initialize_weight('fc7_W'),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005))

            fc2b = tf.get_variable(name='fc2_b', shape=[4096], 
                initializer=self.initialize_weight('fc7_b'))
            fc2l = tf.nn.bias_add(tf.matmul(self.drop1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        self.drop2 = tf.nn.dropout(self.fc2, keep_prob, name='drop2') 

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.random_normal([4096, 8],
                                                         dtype=tf.float32,
                                                         stddev=0.01), name='fc3_W')

            fc3b = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float32),
                                 trainable=True, name='fc3_b')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.drop2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]