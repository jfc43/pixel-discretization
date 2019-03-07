import os
import csv
from scipy.misc import imread
import numpy as np

def load_train_data(data_path):
    images = []
    for filename in open(os.path.join(data_path,'train_sub.txt')):
        filename = filename[:-1]
        image = imread(os.path.join(data_path,filename), mode='RGB').astype(np.float)
        images.append(image)
    return np.array(images)

def load_dev_data(data_path):
  images = [] # images
  labels = [] # corresponding labels
  # loop over all 42 classes
  gtFile = open(os.path.join(data_path,'dev_dataset.csv')) # annotations file
  gtReader = csv.reader(gtFile, delimiter=',') # csv parser for annotations file
  next(gtReader) # skip header
  # loop over all images in current annotations file
  for row in gtReader:
    image = imread(os.path.join(data_path,'images',row[0]+'.png'), mode='RGB').astype(np.float)
    images.append(image) # the 1th column is the filename
    labels.append(int(row[6])) # the 7th column is the label
  gtFile.close()
  return np.array(images), np.array(labels)

def load_test_data(data_path):
  images = [] # images
  labels = [] # corresponding labels
  # loop over all 42 classes
  gtFile = open(os.path.join(data_path,'final_dataset.csv')) # annotations file
  gtReader = csv.reader(gtFile, delimiter=',') # csv parser for annotations file
  next(gtReader) # skip header
  # loop over all images in current annotations file
  for row in gtReader:
    image = imread(os.path.join(data_path,'images_final',row[0]+'.png'), mode='RGB').astype(np.float)
    images.append(image) # the 1th column is the filename
    labels.append(int(row[6])) # the 7th column is the label
  gtFile.close()
  return np.array(images), np.array(labels)
