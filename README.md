# Cirlcle-Detection-DNN
This is a Pytorch implementation of a Deep Convolutional Neural Network model for detecting the parameters of the circle present inside of a given image under the presence of noise.


# Networl Architecture

![Repo List](screenshot/Network.jpg)

The output of the network is 3 real numbers which represents the detected row, column, and radious of the circle in the noisy image. 

# Getting Started

**Installation**
- Clone this repo:
```shell
git clone https://github.com/hsouri/Cirlcle-Detection-DNN
```

- Requirements
Install the dependencies by running the following command:
```shell
pip install -r requirement.txt
```

# Model train/test
- Data set making
Before trainig, train set should be created by the following command:

```shell
python dataset.py
```
This will generate 200,000 images with randon level of noise between 0.035 and 3.5. You can create yout own train set with arbitrary number of imaages and arbitrary level of noise by changing number of images and level of noise in the train_set() function.

- Training:

```shell
python train.py
```
You can use your own data set by changing the default dataset by the following command:

```shell
python train.py -data {directory path to image list}
```

You can also change other attributes such as batch size, learning rate, number of epochs, number of workers, resume
and continue training from a checkpoint. List of selectable attributes:

'-name', '-out_file', '-workers', '-batch-size', '-resume', '-data', '-print_freq', '-epochs', '-start_epoch', '-save_freq'



