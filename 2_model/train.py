import numpy as np
import tensorflow as tf
from datetime import datetime
import random
import argparse
import sys

# dataset iterator
from adience import AdienceIterator
AgeDB = AdienceIterator(0)

# network class
from gilnet import gilnet
#from alexnet import alexnet

# argpraser settings for restore
parser = argparse.ArgumentParser(prog='train.py')
parser = argparse.ArgumentParser()
parser.add_argument('--restore', help='retore a previous training procedure', action="store_true")

args = parser.parse_args()

# global functions for training infos evaluation
def print_iter_info(i, epoch, n_images, acc, loss, accs, losses, accs_one, max_accuracy, acc_one_off=None):
	print("\n"+"-"*30)
	print("Iteration", i,"Epoch", epoch)
	print("-"*30)
	print("No. Images fed:",n_images)
	print("Accuracy：","{:.4f}".format(acc*100)+"%")
	print("Previous Accuracy:","{:.4f}".format(accs[-1]*100)+"%")
	if acc_one_off is not None:
		print("One-off Accuracy:","{:.4f}".format(acc_one_off*100)+"%")
		print("Previous One-off Accuracy:","{:.4f}".format(accs_one[-1]*100)+"%")
	print("Max Test Accuracy:","{:.4f}".format(max_accuracy*100)+"%")
	print("Loss:" ,"{:.6f}".format(loss))
	print("Previous Loss:","{:.6f}".format(losses[-1]))
	print("Loss Difference:","{:.6f}".format(losses[-1]-loss))
	print("-"*30)
	print()

def return_trues(batch_y):
	res = []
	for label in batch_y:
		res.append(label.tolist().index(1))
	return res

def return_accuracy(cls_pred, cls_true):
	res = np.subtract(cls_pred, cls_true)
	res = np.absolute(res).tolist()
	return res.count(0)/len(res), (res.count(0)+res.count(1))/len(res)

# optimizer and loss function definitions
def optimizer(starter_learning_rate, cost):
	global_step = tf.Variable(0, trainable=False)  
	optimizer = tf.train.MomentumOptimizer(starter_learning_rate, 0.9)
	lr_decay_fn = lambda lr, global_step : tf.train.exponential_decay(lr, global_step, 200, 0.97, staircase=True) 
	return tf.contrib.layers.optimize_loss(cost, global_step, starter_learning_rate, 
		optimizer = lambda lr: optimizer, clip_gradients=4., learning_rate_decay_fn=lr_decay_fn) 

def loss(pred, y):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)

	# collect regularization loss from collection
	regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	total_loss = cross_entropy_mean + 0.01 * sum(regularization_losses)

	# apply moving average and add up to total loss
	loss_averages = tf.train.ExponentialMovingAverage(0.9)
	loss_averages_op = loss_averages.apply([cross_entropy_mean] + [total_loss])

	with tf.control_dependencies([loss_averages_op]):
		total_loss = tf.identity(total_loss)
	return total_loss

# network parameters settings
learning_rate = 1e-3
num_epochs = 100
batch_size = 128
dropout = 0.5

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, 8])
keep_prob = tf.placeholder(tf.float32)

# model, cost function and optimizer statements
gilnet = gilnet(x, keep_prob)
logits = gilnet.fc3l
cost = loss(logits, y)
train_op = optimizer(learning_rate, cost)

# Evaluation op: Accuracy of the model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
y_pred_cls = tf.argmax(logits, 1)

# tensorflow savor for variables
# using V1 saver for now
#TODO
#V2 saver cross-platform path bug
saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

# Start Tensorflow session
# Initialize all variables
print("{}: Start training...".format(datetime.now()))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
i = 0

if args.restore:
	saver.restore(sess,"model.ckpt")
	acc_list=np.load("acc_train.npy").tolist()
	loss_list=np.load("loss_train.npy").tolist()
	acc_v_list=np.load("acc_valid.npy").tolist()
	acc_v_one_list = np.load("acc_one_valid.npy").tolist()
	loss_v_list=np.load("loss_valid.npy").tolist()
	validate_i_list=np.load("valid.npy").tolist()
	epoch_list=np.load("epoch.npy").tolist()
	max_accuracy=max(acc_v_list)
	i = (len(loss_list)-1)*10
	print("{}: Model restoring operation success!".format(datetime.now()))

else:
	# Global Training Infos
	# lists and variables keeping track of global training info
	acc_list = [0]
	loss_list = [0]
	acc_v_list = [0]
	acc_v_one_list = [0]
	loss_v_list = [0]
	validate_i_list = [0]
	epoch_list = [1]
	max_accuracy=0

for epoch in range(epoch_list[-1], num_epochs+1):
	# Record Epoch Info
	print("{}: Start Epoch {}...".format(datetime.now(), epoch))
	epoch_list.append(epoch)

	# Training Loop
	batch_x, batch_y, EOF = AgeDB.next_batch(batch_size,"tr")
	n_images = len(batch_x)
	while (not EOF):
		# Stop training after iteration x
		if i==5000:
			print("{}: Done training, exiting...".format(datetime.now(), epoch))
			exit(0)
		
		i += 1

		# Run optimization op (backprop)
		train_dict = {x: batch_x, y: batch_y, keep_prob: dropout}
		sess.run(train_op, feed_dict=train_dict)
		loss, acc = sess.run([cost, accuracy], feed_dict=train_dict)
		batch_x, batch_y, EOF = AgeDB.next_batch(batch_size, 'tr')
		n_images += len(batch_x)

		# print out training info every 10 iterations
		if i%10==0:
			print_iter_info(i, epoch, n_images, acc, loss, acc_list, loss_list, acc_v_one_list, max_accuracy)
			loss_list.append(loss)
			acc_list.append(acc)

			# start a validation loop every 100 iterations or by random
			randnum = random.randint(1, 100)
			if (i%100 == 0) or ((acc>0.55) and (randnum%2==0 or acc_v_list[-1]>0.5)):
				print("{}: Start Testing Iteration {}...".format(datetime.now(), i))
				# prepare for accuracy recording arrays
				num_test = AgeDB.num_test
				cls_pred = np.zeros(shape=num_test, dtype=np.int)
				cls_true = np.zeros(shape=num_test, dtype=np.int)
				m = 0
				n = min(m + batch_size, num_test)
				validate_loss_temp = []

				# start validation
				batch_x_v, batch_y_v, EOF_v = AgeDB.next_batch(batch_size,"va")
				n_images_v = len(batch_x_v)

				# validating loop
				while (not EOF_v):
					validate_dict = {x: batch_x_v, y: batch_y_v, keep_prob: 1.0}
					loss_v, cls_pred[m:n] = sess.run([cost, y_pred_cls], feed_dict=validate_dict)
					validate_loss_temp.append(loss_v)
					cls_true[m:n] = return_trues(batch_y_v)

					batch_x_v, batch_y_v, EOF_v = AgeDB.next_batch(batch_size, 'va')
					n_images_v += len(batch_x_v)
					m = n
					n = min(m + batch_size, num_test)

				# print out validation info
				loss_validate = sum(validate_loss_temp)/len(validate_loss_temp)
				acc_validate, acc_validate_one_off = return_accuracy(cls_pred, cls_true)
				print("{}: Done Testing Iteration {}...".format(datetime.now(), i))
				print_iter_info(i, epoch, n_images_v, acc_validate, loss_validate, acc_v_list, loss_v_list, acc_v_one_list, max_accuracy, acc_validate_one_off)
				loss_v_list.append(loss_validate)
				acc_v_list.append(acc_validate)
				acc_v_one_list.append(acc_validate_one_off)
				validate_i_list.append(i)

				# save model with current maximum validation accuracy
				if acc_v_list[-1]>0.45 and acc_v_list[-1]>max_accuracy:
					save_path = saver.save(sess, "max.ckpt")
					print("Model saved in file: %s" % save_path)
					max_accuracy = acc_v_list[-1]
					np.save("pred.npy", np.array(cls_pred))
					np.save("true.npy", np.array(cls_true))

		if i%200 == 0:
			save_path = saver.save(sess, "model.ckpt")
			print("Model saved in file: %s" % save_path)
			np.save("epoch.npy", np.array(epoch_list))
			np.save("acc_train.npy", np.array(acc_list))
			np.save("loss_train.npy", np.array(loss_list))
			np.save("acc_valid.npy", np.array(acc_v_list))
			np.save("acc_one_valid.npy", np.array(acc_v_one_list))
			np.save("loss_valid.npy", np.array(loss_v_list))
			np.save("valid.npy", np.array(validate_i_list))
