from knnsnn import KnnSnn as ks
from sklearn.neighbors import KDTree

import numpy as np


sample = np.random.rand(20000, 768).astype('float32')

k=20
KSnn = ks(k)

knn_indices = KSnn.knn(sample)
snn_results = KSnn.snn(knn_indices)
