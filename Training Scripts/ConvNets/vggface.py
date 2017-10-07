import tensorflow as tf
import numpy as np

# weights initializers
def vgg_initializer(name, weights=None):
    if weights is not None:
        print(name)
        return tf.constant_initializer(weights[name])
    else:
        if 'b' in name:
            return tf.constant_initializer(0.0)
        else:
            return tf.contrib.keras.initializers.he_normal()

class VGGFace():
    def __init__(self, bgr_mean, weight_file, num_classes=8, weight_decay=5e-4, enable_moving_average=True):
        # input tensors
        self.input_x = tf.placeholder(tf.float32, [None, 224, 224, 3], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        # parameters
        self.parameters = []

        # Try loading pretrained VGG-Face model
        try:
            weights = np.load(weight_file)
        except:
            print("Warning: Pertrained VGG-Face model not found!")
            weights = None

        # zero-mean input
        with tf.name_scope('zero_mean') as scope:
            mean = tf.constant(bgr_mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.input_x-mean
        
        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable(name='conv1_1_W', shape=[3, 3, 3, 64], 
                initializer=vgg_initializer(name='conv1_1_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv1_1_b', shape=[64], 
                initializer=vgg_initializer(name='conv1_1_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.get_variable(name='conv1_2_W', shape=[3, 3, 64, 64], 
                initializer=vgg_initializer(name='conv1_2_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv1_2_b', shape=[64], 
                initializer=vgg_initializer(name='conv1_2_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                                    padding='SAME', name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.get_variable(name='conv2_1_W', shape=[3, 3, 64, 128], 
                initializer=vgg_initializer(name='conv2_1_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))         
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv2_1_b', shape=[128], 
                initializer=vgg_initializer(name='conv2_1_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            kernel = tf.get_variable(name='conv2_2_W', shape=[3, 3, 128, 128], 
                initializer=vgg_initializer(name='conv2_2_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv2_2_b', shape=[128], 
                initializer=vgg_initializer(name='conv2_2_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool2')
        
        # conv3_1
        with tf.variable_scope('conv3_1') as scope:
            kernel = tf.get_variable(name='conv3_1_W', shape=[3, 3, 128, 256], 
                initializer=vgg_initializer(name='conv3_1_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv3_1_b', shape=[256], 
                initializer=vgg_initializer(name='conv3_1_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            kernel = tf.get_variable(name='conv3_2_W', shape=[3, 3, 256, 256], 
                initializer=vgg_initializer(name='conv3_2_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv3_2_b', shape=[256], 
                initializer=vgg_initializer(name='conv3_2_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            kernel = tf.get_variable(name='conv3_3_W', shape=[3, 3, 256, 256], 
                initializer=vgg_initializer(name='conv3_3_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv3_3_b', shape=[256], 
                initializer=vgg_initializer(name='conv3_3_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
            kernel = tf.get_variable(name='conv4_1_W', shape=[3, 3, 256, 512], 
                initializer=vgg_initializer(name='conv4_1_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv4_1_b', shape=[512], 
                initializer=vgg_initializer(name='conv4_1_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            kernel = tf.get_variable(name='conv4_2_W', shape=[3, 3, 512, 512], 
                initializer=vgg_initializer(name='conv4_2_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv4_2_b', shape=[512], 
                initializer=vgg_initializer(name='conv4_2_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            kernel = tf.get_variable(name='conv4_3_W', shape=[3, 3, 512, 512], 
                initializer=vgg_initializer(name='conv4_3_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv4_3_b', shape=[512], 
                initializer=vgg_initializer(name='conv4_3_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1') as scope:
            kernel = tf.get_variable(name='conv5_1_W', shape=[3, 3, 512, 512], 
                initializer=vgg_initializer(name='conv5_1_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv5_1_b', shape=[512], 
                initializer=vgg_initializer(name='conv5_1_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
            kernel = tf.get_variable(name='conv5_2_W', shape=[3, 3, 512, 512], 
                initializer=vgg_initializer(name='conv5_2_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv5_2_b', shape=[512], 
                initializer=vgg_initializer(name='conv5_2_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            kernel = tf.get_variable(name='conv5_3_W', shape=[3, 3, 512, 512], 
                initializer=vgg_initializer(name='conv5_3_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable(name='conv5_3_b', shape=[512], 
                initializer=vgg_initializer(name='conv5_3_b', weights=weights))
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name='conv')
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name='pool5')
                                    
        shape = int(np.prod(self.pool5.get_shape()[1:]))
        flat = tf.reshape(self.pool5, [-1, shape])
                                    
        # fc6
        with tf.variable_scope('fc6'):
            w = tf.get_variable('fc6_W', [shape, 4096], initializer=vgg_initializer(name='fc6_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            b = tf.get_variable('fc6_b', [4096], initializer=vgg_initializer(name='fc6_b', weights=weights))
            out = tf.matmul(flat, w) + b
            self.fc6 = tf.nn.relu(out)
            self.drop1 = tf.nn.dropout(self.fc6, self.dropout_keep_prob, name='drop1')
            self.parameters += [w, b]

        # fc7
        with tf.variable_scope('fc7'):
            w = tf.get_variable('fc7_W', [self.drop1.get_shape()[1], 4096], initializer=vgg_initializer(name='fc7_W', weights=weights),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            b = tf.get_variable('fc7_b', [4096], initializer=vgg_initializer(name='fc7_b', weights=weights))
            out = tf.matmul(self.drop1, w) + b
            self.fc7 = tf.nn.relu(out)
            self.drop2 = tf.nn.dropout(self.fc7, self.dropout_keep_prob, name='drop2')
            self.parameters += [w, b]

        # fc8
        with tf.variable_scope('fc8'):
            w = tf.get_variable('fc8_W', [self.drop2.get_shape()[1], num_classes], 
                initializer=tf.random_normal_initializer(stddev=0.01),
                regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
            b = tf.get_variable('fc8_b', [num_classes], initializer=tf.constant_initializer(0.0))
            self.fc8 = tf.matmul(self.drop2, w) + b
            self.parameters += [w, b]

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            self.predictions = tf.argmax(self.fc8, 1, name="predictions")
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.fc8, labels = self.input_y)
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