from knnsnn import KnnSnn as ks
from sklearn.neighbors import KDTree

import numpy as np

sample = np.random.rand(100, 768).astype('float32')

k=20
KSnn = ks()

indices = KSnn.knn(sample)

print(indices[3])


tree = KDTree(sample)  
_, ind_sk = tree.query(sample, k=k)

print(ind_sk[3])
