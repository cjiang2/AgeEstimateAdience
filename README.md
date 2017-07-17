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

## Usage

### Preprocessing

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


### CNN Network

Three networks architectures implementations are provided: GilNet, AlexNet and VGG-16. Check out papers for to see the architectures for each CNN:

	- GilNet: http://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf

	- AlexNet: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf

	- VGG-16: https://arxiv.org/pdf/1409.1556.pdf

Two versions are provided for each CNN: raw implementations and tensorflow.contrib.layers implementations(Will be updated after I finished up clean implementations and testing for each of them). Use whatever version you like since the final result does not hold any significant difference(at least from my testing). But it is noted that contrib.layers implementations seems to require more computational power than raw implementation.


### Training Protocols

Fine-tuning of the Network:

	While the weights of GilNet and AlexNet are initialized with random normal variables, it is noted that weights initializing of a network(especially a large one like VGG) with random variables might not be a good idea. In the implementation of VGG-16 network, I'm finetuning weights from [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) models.

	While the link does not provide a direct tensorflow model, however it does provide a Caffe version of weights file, and I'm recommanding [Caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) to convert weights into raw numpy files, which can be applied into tensorflow directly using constant initializer or assign operation. (See my [implementation of VGG-16 network](https://github.com/zonetrooper32/AgeEstimateAdience/blob/master/2_model/vgg16.py) to understand how to initialze weights with numpy arrays)

Training Details:

	-_- I'll fill this up after I finish up clean implementation version

### Real-Time Prediction




## Reference
