import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import json

def KDEProximate(points, r, k):
    n = points.shape[0]
    mask = np.ones(n)
    codes = []
    hs = {}
    for i in range(n):
        if not points[i,0] in hs:
            hs[points[i,0]] = 0
        else:
            hs[points[i,0]] += 1
    score = np.zeros(n)
    for i in range(n):
    	score[i] = hs[points[i,0]]
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

mnist = input_data.read_data_sets('data/fashion', one_hot=False)

images = mnist.train.images.astype(float)
points = images.reshape((-1,1))
codes = KDEProximate(points,r=r,k=k)

print('# codes: %d'%codes.shape[0])
np.save(codes_path,codes)