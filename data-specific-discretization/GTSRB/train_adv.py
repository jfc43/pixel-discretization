"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from util import preprocess

from model import Model
import gtsrb_input
from pgd_attack import LinfPGDAttack
from CW_attack import CWAttack
import re

with open('config.json') as config_file:
  config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

epsilon = config['epsilon']
model_dir = config['model_dir']
base_model_dir = config['base_model_dir']
use_pretrain = config['use_pretrain']
step_size = config['step_size']
alpha = config['alpha']
attack_steps = config['attack_steps']
random_start = config['random_start']
loss_func = config['loss_func']
codes_path = config['codes_path']
discretize = config['discretize']

if discretize:
  codes = np.load(codes_path)

# Setting up the data and the model
raw_cifar = gtsrb_input.GTSRBData(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=5)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv', model.xent / batch_size)
#tf.summary.image('images adv train', model.x_input)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth=True

with tf.Session(config = tf_config) as sess:
  # initialize data augmentation
  cifar = gtsrb_input.AugmentedGTSRBData(raw_cifar, sess, model)

  # Initialize the summary writer, global variables, and our time counter.
  train_summary_writer = tf.summary.FileWriter(os.path.join(model_dir,'train'), sess.graph)
  test_summary_writer = tf.summary.Filewriter(os.path.join(model_dir,'test'), sess.graph)

  if use_pretrain:
    checkpoint = tf.train.latest_checkpoint(base_model_dir)
    saver.restore(sess, checkpoint)
    curr_step = int(checkpoint.split('-')[1])
    sess.run(global_step.assign(curr_step))
  else:
    curr_step = 0
    sess.run(tf.global_variables_initializer())

  training_time = 0.0

  # Set up adversary
  if discretize:
    attack = CWAttack(model, attack_steps, step_size, epsilon, codes, batch_size, alpha)
  else:
    attack = LinfPGDAttack(model, epsilon, attack_steps, step_size, random_start, loss_func)

  # Main training loop
  for ii in range(curr_step, max_num_training_steps):
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                       multiple_passes=True)
    # Compute Adversarial Perturbations
    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    end = timer()

    if discretize:
      x_batch_ = preprocess(x_batch, codes)
      x_batch_adv_ = preprocess(x_batch_adv, codes)
    else:
      x_batch_ = x_batch
      x_batch_adv_ = x_batch_adv

    training_time += end - start

    nat_dict = {model.x_input: x_batch_,
                model.y_input: y_batch}

    adv_dict = {model.x_input: x_batch_adv_,
                 model.y_input: y_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      nat_loss = sess.run(model.xent, feed_dict=nat_dict)
      adv_loss = sess.run(model.xent, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%, loss {:.4}'.format(nat_acc * 100,nat_loss))
      print('    training adv accuracy {:.4}%, loss {:.4}'.format(adv_acc * 100,adv_loss))
      if ii != curr_step:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=adv_dict)
      train_summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=adv_dict)
    end = timer()
    training_time += end - start
