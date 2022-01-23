'''
HOWTOUSE for knnsnn.py
'''


from knnsnn import KnnSnn as ks ## import
import numpy as np


sample = np.random.rand(20000, 768).astype('float32')

k=20
KSnn = ks(k) ## initialize the instance

knn_indices = KSnn.knn(sample) ## run knn
snn_results = KSnn.snn(knn_indices) ## run snn
