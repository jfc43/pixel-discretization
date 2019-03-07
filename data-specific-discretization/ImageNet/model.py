from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2
slim = tf.contrib.slim

class Model(object):
  """ResNet model."""

  def __init__(self, method):
    self.x_input = tf.placeholder(
        tf.float32,
        shape=[None, 299, 299, 3])

    self.method = method
    self.y_input = tf.placeholder(tf.int64, shape=None)

    self.pre_softmax = self.forward(self.x_input)

    self.softmax = tf.nn.softmax(self.pre_softmax)

    self.predictions = tf.argmax(self.pre_softmax, 1)
    self.correct_prediction = tf.equal(self.predictions, self.y_input)
    self.num_correct = tf.reduce_sum(
        tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))

    self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.pre_softmax, labels=self.y_input)
    self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
    self.mean_xent = tf.reduce_mean(self.y_xent)


  def forward(self, x_input):
    x_input = (x_input * 2./255) - 1
    if self.method == 'nat':
        with slim.arg_scope(inception.inception_v3_arg_scope()):
          _, end_points = inception.inception_v3(x_input, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    else:
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(x_input, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
          
    return end_points['Logits']
