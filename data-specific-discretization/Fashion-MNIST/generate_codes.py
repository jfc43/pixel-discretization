import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import json

with open('config.json') as config_file:
  config = json.load(config_file)

codes_path = config['codes_path']
cluster_algorithm = config['cluster_algorithm']

if cluster_algorithm == 'KM':
    import matlab.engine
    eng = matlab.engine.start_matlab('-nodisplay')

def KM(points, k):
    idx, C = eng.kmedoids(matlab.double(points.tolist()),k,'Distance','cityblock',nargout=2)
    return np.array(C)

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

mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=False)

images = mnist.train.images.astype(float)
points = images.reshape((-1,1))

if cluster_algorithm == 'KDE':
    k = config['k']
    r = config['r']
    codes = KDEProximate(points,r=r,k=k)
elif cluster_algorithm == 'KM':
    k = config['k']
    np.random.shuffle(points)
    points = points[0:1000000]
    codes = KM(points, k)
else:
    print('Not supported clustering algorithm')

np.save(codes_path,codes)
