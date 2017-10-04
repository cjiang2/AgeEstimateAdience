import numpy as np
import random
import h5py

def load_dataset(dataset_file, validate_folder_no):
	train_data = []
	train_label = []
	with h5py.File(dataset_file, 'r') as hf:
		for i in range(1, 6):
			if i==validate_folder_no:
				test_data = hf["data_"+str(i)].value
				test_label = hf["labels_"+str(i)].value
			else:
				train_data.append(hf["data_"+str(i)].value)
				train_label.append(hf["labels_"+str(i)].value)
		bgr_means = hf["bgr_means"].value.tolist()
	return np.concatenate(train_data), np.concatenate(train_label), test_data, test_label, bgr_means[validate_folder_no - 1]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]