import cv2
import numpy as np
import os
import random

class AdienceIterator:

	def __init__(self, k):
		self.k = str(k)
		self.training = []
		self.validate = []
		self._init_k_folds()
		self._init_training()
		self._init_validate()

	def _init_k_folds(self):
		self.k_folds = "0 1 2 3 4".split()
		self.k_folds.remove(self.k)

	def _init_training(self):
		self.training = []
		for k_fold in self.k_folds:
			cwd = os.getcwd()+"/"+k_fold
			for root, dirs, filenames in os.walk(cwd):
				for i in range(0, len(filenames)):
					filenames[i] = cwd+"/"+filenames[i]
				self.training += filenames
		random.shuffle(self.training)

	def _init_validate(self):
		self.validate = []
		cwd = os.getcwd()+"/"+self.k
		for root, dirs, filenames in os.walk(cwd):
			for i in range(0, len(filenames)):
				filenames[i] = cwd+"/"+filenames[i]
			self.validate += filenames
		random.shuffle(self.validate)

	def _init_next_epoch(self, flag):
		if flag=="tr":
			self._init_training()
			random.shuffle(self.training)
		else:
			self._init_validate()
			random.shuffle(self.validate)

	# channel-wise mean subtraction
	# calculate RGB mean with mean_image.py
	def _preprocess(self, img):
		img = img.astype(np.float64)
		img[:,:,0] -= 116.96
		img[:,:,1] -= 89.83
		img[:,:,2] -= 80.18
		return img

	def next_batch(self, batch_size=64, flag="tr"):
		if flag=="tr":
			images = self.training
		else:
			images = self.validate

		if len(images) < batch_size:
			if len(images) == 0:
				self._init_next_epoch(flag)
				return [], [], True
			else:
				batch_len = len(images)
		else:
			batch_len = batch_size

		batch_x = []
		batch_y = []
		for i in range(0, batch_len):
			img = cv2.imread(images[i])
			img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			img = self._preprocess(img)
			batch_x.append(img)
			batch_y.append(self.return_label(images[i]))

		if flag == 'tr':
			self.training = self.training[batch_len:]
		else:
			self.validate = self.validate[batch_len:]

		return batch_x, batch_y, False

	def return_label(self, image):
		token = image.split("_")
		index = int(token[0][-1])
		res = []
		res = np.zeros(8)
		res[index] = 1
		return res