import numpy as np
import json

with open('config.json') as config_file:
  config = json.load(config_file)

k = config['k']
codes_path = config['codes_path']

codes = []
for i in range(k):
    codes.append([float(i)/(k-1)*255])
codes = np.array(codes)

print('# codes: %d'%codes.shape[0])
np.save(codes_path,codes)
