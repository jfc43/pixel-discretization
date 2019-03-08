# Towards Understanding Limitations of Pixel Discretization Against Adversarial Attacks
This project is for the paper [Towards Understanding Limitations of Pixel Discretization Against Adversarial Attacks](https://arxiv.org/pdf/1805.07816.pdf). Some codes are from [MNIST Challenge](https://github.com/MadryLab/mnist_challenge) and [CIFAR10 Challenge](https://github.com/MadryLab/cifar10_challenge). 

## Preliminaries
It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requries tensorflow package to be installed:
* [Tensorflow](https://www.tensorflow.org/install)
* [scipy](https://github.com/scipy/scipy)
* [sklearn](https://scikit-learn.org/stable/)
* [numpy](http://www.numpy.org/)
* [matlab-for-python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html)

## Downloading Datasets
* [MNIST](http://yann.lecun.com/exdb/mnist/): included in Tensorflow. 
* [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): included in Tensorflow.
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): unpickles the CIFAR10 dataset from a specified folder containing a pickled version following the format of Krizhevsky. 
* [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset): download the training dataset and test dataset from the [website](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 
* [ImageNet](https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack/data): scripts are provided to download the dataset.

## Downloading Pre-trained Models
* MNIST: use fetch_model.py in the folder to download pre-trained models.
* CIFAR-10: use fetch_model.py in the folder to download pre-trained models.
* ImageNet: use download_checkpoint.sh to download pre-trained models. 

## Running Experiments
* Before doing experiments, first edit config.json file to specify experiment settings.
* generate_codes.py: generate codes for discretization. 
* train_nat.py: naturally train models.
* train_adv.py: adversarially train models. 
* eval.py: evaluate the models trained. 
