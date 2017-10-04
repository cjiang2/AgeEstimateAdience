# AgeEstimateAdience
Age and Gender Estimation Using Convolutional Neural Network with Adience Dataset

## Note
This project will focus on age and gender prediction using Adience dataset. The purpose of this project is to construct and demo popular CNN network architectures using Tensorflow with image classification task.

(Currently I'm updating a improved version of training script.) Refer to [another branch](https://github.com/zonetrooper32/AgeEstimateAdience/tree/old_scripts) to see old training scripts.

## Prerequisites

 - Python3
 - Tensorflow 1.0 or higher
 - OpenCV
 - Jupyter Notebook
 - h5py

## Getting Started

To Download the Entire Adience Dataset:
```
http://www.openu.ac.il/home/hassner/Adience/data.html
```

## Preprocessing

 -  Make_Dataset_For_Aligned.ipynb  Jupyter Notebook used to compress the entire Adience Dataset(aligned.tar.gz version)
 
 -  Make_Dataset_For_Faces.ipynb  Jupyter Notebook used to compress the entire Adience Dataset(faces.tar.gz version, which contains landmark infos)

 Note: Both Jupyter Notebooks use BGR mode for image reading and contain channel-wise mean calculation and mean image calculation. The train script uses channel-wise mean subtraction as preprocessing technique. 
 
 While both channel-wise mean subtraction and mean image subtraction are two of the popular preprocessing techniques, I would suggest to try channel-wise mean subtraction first if the backgrounds of the images in your dataset have a lot of variations. If preprocessing images from the source like a static camera(which has a high chance of "common background"), mean image is the way to go. 
 
 It should also be noted that VGG-Face model uses a BGR mean of [93.5940, 104.7624, 129.1863] for channel-wise mean subtraction, which is calculated originally from VGG-Face Dataset.

## CNN Network

Three networks architectures implementations are provided: GilNet, AlexNet and VGG-16. Check out papers for to see the architectures for each CNN:

	- GilNet: http://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf

	- AlexNet: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf

 	- VGG-Face: http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf

## Training

  ### Local Adience Database I/O Iterator
  
  - data_helper.py A simple script used for h5py database iteration, batch generation.

  ### Training and Testing

  - train.py A simple script for model training and testing

  ```
	usage: train.py [-h] [--dataset_file DATASET_FILE]
			[--folder_to_test FOLDER_TO_TEST]
			[--dropout_keep_prob DROPOUT_KEEP_PROB]
			[--weight_decay WEIGHT_DECAY] [--learning_rate LEARNING_RATE]
			[--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
			[--evaluate_every EVALUATE_EVERY]
			[--enable_moving_average [ENABLE_MOVING_AVERAGE]]
			[--noenable_moving_average]

	optional arguments:
	  -h, --help            show this help message and exit
	  --dataset_file DATASET_FILE
				Path for the h5py dataset.
	  --folder_to_test FOLDER_TO_TEST
				No. of folder to be tested (default: 1)
	  --dropout_keep_prob DROPOUT_KEEP_PROB
				Dropout keep probability (default: 0.5)
	  --weight_decay WEIGHT_DECAY
				Weight decay rate for L2 regularization (default:
				5e-4)
	  --learning_rate LEARNING_RATE
				Starter Learning Rate (default: 1e-2)
	  --batch_size BATCH_SIZE
				Batch Size (default: 128)
	  --num_epochs NUM_EPOCHS
				Number of training epochs (default: 200)
	  --evaluate_every EVALUATE_EVERY
				Evaluate model on dev set after this many steps
				(default: 50)
	  --enable_moving_average [ENABLE_MOVING_AVERAGE]
				Enable usage of Exponential Moving Average (default:
				True)
	  --noenable_moving_average
  ```

## Real-Time Prediction

  -_- Filling this place later

## Reference

  [New Train Script and data iterator from Convolutional Neural Network for Text Classification](https://github.com/dennybritz/cnn-text-classification-tf)

  [Image Dataset Compression](https://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN/blob/master/celeba_make_dataset.ipynb)
  
  [Pre-trained VGG-Face Model](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
  
  [Original Implementation of VGG-16 ConvNet](https://www.cs.toronto.edu/~frossard/post/vgg16/)

