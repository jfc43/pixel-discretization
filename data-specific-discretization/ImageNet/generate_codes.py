import numpy as np
import os
from input_data import *

def KDEProximate(points, r, k):
    n = points.shape[0]
    mask = np.ones(n)
    codes = []
    hs = np.zeros((256,256,256))
    for i in range(n):
        hs[int(points[i,0]),int(points[i,1]),int(points[i,2])] += 1
    score = np.zeros(n)
    for i in range(n):
    	score[i] = hs[int(points[i,0]),int(points[i,1]),int(points[i,2])]
    for i in range(k):
        if not any(mask):
            break
        j = np.argmax(score*mask)
        codes.append(points[j])
        d = np.linalg.norm(points-points[j],ord=np.inf,axis=1)
        mask[np.where(d<=r)] = 0
    return np.array(codes)

with open('config.json') as config_file:
  config = json.load(config_file)

k = config['k']
r = config['r']
codes_path = config['codes_path']
data_path = config['data_path']

images, labels = load_dev_data(data_path)
images = images.astype(float)
points = images.reshape((-1,3))
codes = KDEProximate(points,r=r,k=k)
print('# global codes: %d'%codes.shape[0])
np.save(codes_path, codes)
