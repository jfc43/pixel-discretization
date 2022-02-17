# Towards Understanding Limitations of Pixel Discretization Against Adversarial Attacks
This project is for the paper [Towards Understanding Limitations of Pixel Discretization Against Adversarial Attacks](https://arxiv.org/pdf/1805.07816.pdf). Some codes are from [MNIST Challenge](https://github.com/MadryLab/mnist_challenge) and [CIFAR10 Challenge](https://github.com/MadryLab/cifar10_challenge). 

## Preliminaries
It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requires tensorflow package to be installed:
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

## Overview of the Code
### Running Experiments
* Before doing experiments, first edit config.json file to specify experiment settings.
* generate_codes.py: generate codes for discretization. 
* train_nat.py: naturally train models.
* train_adv.py: adversarially train models. 
* eval.py: evaluate the models trained. 
### Parameters in `config.json`
Model configuration:
- `model_dir`: contains the path to the directory of the currently trained/evaluated model.

GPU configuration:
- `gpu_device`: which gpu device to use. Should be a string.

Data configuration:
- `data_path`: contains the path to the directory of dataset. 

Training configuration:
- `tf_random_seed`: the seed for the RNG used to initialize the network weights.
- `numpy_random_seed`: the seed for the RNG used to pass over the dataset in random order.
- `max_num_training_steps`: the number of training steps.
- `num_output_steps`: the number of training steps between printing progress in standard output.
- `num_summary_steps`: the number of training steps between storing tensorboard summaries.
- `num_checkpoint_steps`: the number of training steps between storing model checkpoints.
- `training_batch_size`: the size of the training batch.
- `use_pretrain`: use pretrained model or not. Can be `true` or `false`.
- `base_model_dir`: contains the path to the directory of pretrained model. 

Evaluation configuration:
- `num_eval_examples`: the number of CIFAR10 examples to evaluate the model on.
- `eval_batch_size`: the size of the evaluation batches.

Adversarial examples configuration:
- `epsilon`: the maximum allowed perturbation per pixel.
- `attack_steps`: the number of PGD iterations used by the adversary.
- `step_size`: the size of the PGD adversary steps.
- `random_start`: specifies whether the adversary will start iterating from the natural example or a random perturbation of it.
- `loss_func`: the loss function used to run pgd on. `xent` corresponds to the standard cross-entropy loss, `cw` corresponds to the loss function of [Carlini and Wagner](https://arxiv.org/abs/1608.04644).
- `store_adv_path`: the file in which adversarial examples are stored. Relevant for the `pgd_attack.py` and `run_attack.py` scripts.
- `alpha`: the hyper-parameter in the discretization approximation function. 

Discretization configuration:
- `codes_path`: the path to the `*.npy` file saving the codes generated.
- `cluster_algorithm`: the algorithm used to find the codes. Can be `KDE` or `KM`.
- `discretize`: use discretization or not. Can be `true` or `false`.
- `k`: number of codes.
- `r`: the minimum distance between two codes. Used in the data-specific discretization. 

## Example usage
After cloning the repository you can either train a new network or evaluate/attack one of the pre-trained networks.

#### Training a new network
* Start training by running:
```
python train.py
```

#### Test the network
* Evaluate the model trained by running:
```
python eval.py
```

### Citation 
Please cite our work if you use the codebase: 
```
@inproceedings{chen2019towards,
  title={Towards understanding limitations of pixel discretization against adversarial attacks},
  author={Chen, Jiefeng and Wu, Xi and Rastogi, Vaibhav and Liang, Yingyu and Jha, Somesh},
  booktitle={2019 IEEE European Symposium on Security and Privacy (EuroS\&P)},
  pages={480--495},
  year={2019},
  organization={IEEE}
}
```

### License
Please refer to the [LICENSE](LICENSE).
