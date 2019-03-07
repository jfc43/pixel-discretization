from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import csv
import matplotlib.pyplot as plt
from PIL import Image
version = sys.version_info

import numpy as np
import skimage.data
import scipy.io as sio

class GTSRBData(object):
    def __init__(self, path):
        train_images, train_labels = self.load_training_data('data/Training/Images')
        eval_images, eval_labels = self.load_testing_data('data/Testing/Images')

        self.train_data = DataSubset(train_images, train_labels)
        self.eval_data = DataSubset(eval_images, eval_labels)

    @staticmethod
    def load_testing_data(rootpath):
        images = [] # images
        labels = [] # corresponding labels
        prefix = rootpath + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-final_test.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            img = np.array(Image.fromarray(plt.imread(prefix + row[0])).resize((32,32)))
            intensity = np.mean(img.reshape((-1)))
            if intensity <50:
               continue
            images.append(img) # the 1th column is the filename
            labels.append(int(row[7])) # the 8th column is the label
        gtFile.close()
        return np.array(images), np.array(labels)

    @staticmethod
    def load_training_data(rootpath):
        images = [] # images
        labels = [] # corresponding labels
        # loop over all 42 classes
        for c in range(0,43):
            prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
            gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader) # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                img = np.array(Image.fromarray(plt.imread(prefix + row[0])).resize((32,32)))
                if intensity <50:
                    continue
                images.append(img) # the 1th column is the filename
                labels.append(int(row[7])) # the 8th column is the label
            gtFile.close()
        return np.array(images), np.array(labels)


class AugmentedGTSRBData(object):
    def __init__(self, raw_GTSRBdata, sess, model):
        assert isinstance(raw_GTSRBdata, GTSRBData)
        self.image_size = 32

        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
            img, self.image_size + 4, self.image_size + 4),
            self.x_input_placeholder)
        cropped = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size,
                                                             self.image_size,
                                                             3]), padded)
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
        self.augmented = flipped

        self.train_data = AugmentedDataSubset(raw_GTSRBdata.train_data, sess,
                                             self.x_input_placeholder,
                                              self.augmented)
        self.eval_data = AugmentedDataSubset(raw_GTSRBdata.eval_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)


class DataSubset(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += actual_batch_size
        return batch_xs, batch_ys


class AugmentedDataSubset(object):
    def __init__(self, raw_datasubset, sess, x_input_placeholder,
                 augmented):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        raw_batch = self.raw_datasubset.get_next_batch(batch_size, multiple_passes,
                                                       reshuffle_after_pass)
        images = raw_batch[0].astype(np.float32)
        return self.sess.run(self.augmented, feed_dict={self.x_input_placeholder:
                                                    raw_batch[0]}), raw_batch[1]
