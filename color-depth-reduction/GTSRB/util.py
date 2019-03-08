import numpy as np
from sklearn.neighbors import KDTree

def preprocess(images0, codes):
  images = np.copy(images0)
  kd = KDTree(codes, metric='infinity')
  new_images = []
  for img in images:
    points = img.reshape(-1,1)
    inds = np.squeeze(kd.query(points,return_distance=False))
    new_images.append(codes[inds].reshape(img.shape))
  return np.array(new_images)
