import numpy as np
import math
import cv2
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import json

from util import preprocess
from model import Model
from CW_attack import CWAttack

with open('config.json') as config_file:
  config = json.load(config_file)

epsilon = config['epsilon']
model_dir = config['model_dir']
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']
step_size = config['step_size']
alpha = config['alpha']
attack_steps = config['attack_steps']
random_start = config['random_start']
loss_func = config['loss_func']
codes_path = config['codes_path']
discretize = config['discretize']
gpu_device = config['gpu_device']

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

if discretize:
  codes = np.load(codes_path)

if __name__=='__main__':
  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    model = Model()
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, checkpoint)

    if discretize:
      attack = CWAttack(model, attack_steps, step_size, epsilon, codes, eval_batch_size, alpha)
    else:
      attack = LinfPGDAttack(model, epsilon, attack_steps, step_size, random_start, loss_func)

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_corr_nat = 0
    total_corr_adv = 0
    rate = []

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      dict_nat = {model.x_input: x_batch_,
                  model.y_input: y_batch}

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      if discretize:
        x_batch_ = preprocess(x_batch, codes)
        x_batch_adv_ = preprocess(x_batch_adv, codes)
      else:
        x_batch_ = x_batch
        x_batch_adv_ = x_batch_adv

      dict_adv = {model.x_input: x_batch_adv_,
                    model.y_input: y_batch}

      cur_corr_nat = sess.run(
                        model.num_correct,
                        feed_dict = dict_nat)
      cur_corr_adv = sess.run(
                        model.num_correct,
                        feed_dict = dict_adv)

      print('batch: %d/%d'%(ibatch+1,num_batches))
      print("Correctly classified natural examples: {}".format(cur_corr_nat))
      print("Correctly classified adversarial examples: {}".format(cur_corr_adv))

      total_corr_nat += cur_corr_nat
      total_corr_adv += cur_corr_adv

    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('adversarial: {:.2f}%'.format(100 * acc_adv))
