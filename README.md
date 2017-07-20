# AgeEstimateAdience
Age and Gender Estimation Using Convolutional Neural Network with Adience Dataset

## Note
This project will focus on age and gender prediction using Adience dataset. The purpose of this project is to construct and demo popular CNN network architectures using Tensorflow with image classification task.

## Prerequisites

 - Python3
 - Tensorflow 1.0 or higher
 - OpenCV

## Getting Started

To Download the Entire Adience Dataset:
```
http://www.openu.ac.il/home/hassner/Adience/data.html
```

## Preprocessing

 - mean_image.py Tool used to calculate either a mean image or BGR channels mean.

```
usage: compute_mean_image.py [-h] {bgr,img} ...

positional arguments:
  {bgr,img}
    bgr       generate three channels mean [compute_mean_image.py' -bgr -h]
    img       generate mean image [compute_mean_image.py' -img -h]

optional arguments:
  -h, --help  show this help message and exit
```

Note: While both channel-wise mean subtraction and mean image subtraction are two of the popular preprocessing techniques, I would suggest to try channel-wise mean subtraction first if the backgrounds of the images in your dataset have a lot of variations. If preprocessing images from the source like a static camera(which has a high chance of "common background"), mean image is the way to go. 


## CNN Network

Three networks architectures implementations are provided: GilNet, AlexNet and VGG-16. Check out papers for to see the architectures for each CNN:

	- GilNet: http://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf

	- AlexNet: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf

	- VGG-16: https://arxiv.org/pdf/1409.1556.pdf

Two versions are provided for each CNN: raw implementations and tensorflow.contrib.layers implementations(Will be updated after I finished up clean implementations and testing for each of them). Use whatever version you like since the final result does not hold any significant difference(at least from my testing). But it is noted that contrib.layers implementations seems to require more computational power than raw implementation.


## Training Protocols

## Fine-tuning of the Network:

  While the weights of GilNet and AlexNet are initialized with random normal variables, it is noted that weights initializing of a network(especially a large one like VGG) with random variables might not be a good idea. In the implementation of VGG-16 network, I'm finetuning weights from [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) models.

  While the link does not provide a direct tensorflow model, however it does provide a Caffe version of the weights file, and I'm recommanding [Caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) to convert weights into raw numpy files, and those numpy files can be applied into tensorflow directly using constant initializer or assign operation. (See my [implementation of VGG-16 network](https://github.com/zonetrooper32/AgeEstimateAdience/blob/master/2_model/vgg16.py) to understand how to initialze weights with numpy arrays)

  I'm also providing a simple script named extract_weights.py which loads the converted numpy binary file from Caffe-tensorflow and extract weights and biases for each single layer.

## Training Details:

  ### Local Adience Database I/O Iterator
  
  - adience.py A simple script used for database iteration, batch generation.

  ```
  usage example:

    from adience import AdienceIterator
    iterator = AdienceIterator(cross_validation_folder_no=0)

  Note:

    cross_validation_folder_no: folder number 0~4

  ```

  While a simple I/O script with imread operations of OpenCV can get the job done, using h5py is a good low-cost method. Here's an [example](https://github.com/timsainb/Tensorflow-MultiGPU-VAE-GAN/blob/master/celeba_make_dataset.ipynb) of how to compress an image dataset into a single h5py file.

  ### Training and Testing

  - train.py A simple script for model training and testing

  ```
  usage: train.py [-h] [--restore]

  optional arguments:
    -h, --help  show this help message and exit
    --restore   retore a previous training procedure
  ```

    Training procedure is done with a Momentum Optimizer. You can also use AdamOptimizer or SGD. I'm also applying Weight Regularization, learning rate decay and Exponential Moving Average operation during the training process.

    Training script will print out relevant training infos every 10 iterations. A model validation will be done every 100 iteration or by random(To make sure we get the model with best test accuracy).  Every 200 iterations, a tensorflow V1 format copy of the model along with iteration infos will be saved locally. The script will also be automatically saving model if the test accuracy is high enough. See the implementation to understand how training is done in general. 

  ### Relevant Statistics

    Since it is easy to extract weights or layers given my implementation of CNN network class, you can easily monitor the overall training procedure. For starters, I'm recommanding [this](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb) tutorials which gives a detailed introduction on acquiring statistics including confusion matrix, layers outputs and etc. I'm also recommanding using TensorBoard.

## Real-Time Prediction

  - train.py A simple script for model training and testing


## Reference
