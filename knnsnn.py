from numba import cuda
import faiss
import numpy as np
import math

@cuda.jit
def snn_cuda(raw_knn, snn_result, k_list, length_list):
	## Input: raw_knn (knn info)
	## Output: snn_strength (snn info)
	k = k_list[0]
	length = length_list[0]

	i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
	j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

	if i >= length or j >= length:
			return
	if i == j:
			snn_result[i, j] = 0
			return
	c = 0
	for m in range(k):
			for n in range(k):
					if raw_knn[i, m] == raw_knn[j,n]:
							c += (k + 1 - m) * (k + 1 - n)

	snn_result[i, j] = c



class KnnSnn:

	'''
	Initialize the KNNSNN class instance
	desingate k value which will be used throughout the computation of knn and SNN
	'''
	def __init__(self, k=20):
		self.k=k

	'''
	INPUT
	- data: 2D numpy array (N, D) where 
	  - N denotes the number of data points 
		- D denotes the dimensionality
	'''
	def knn(self, data):

		index = faiss.IndexFlatL2(data.shape[1])
		index.add(data)

		_, indices = index.search(data, self.k + 1)
		return indices[:, 1:]

	'''
	INPUT
	- raw_knn: output of knn() defined above: (N, self.k) array
	 - N denotes the number of data points
	 - self.k denotes the k value for SNN computation (given in prior)
	- TPB: threads_per_block value (default=16)

	- Note that resulting SNN matrix is not normalized!!
	'''
	def snn(self, raw_knn, TPB=16):

		raw_knn_global_mem    = cuda.to_device(np.ascontiguousarray(raw_knn))
		length                = len(raw_knn)
		snn_result_global_mem = cuda.device_array((length, length))

		length_list_global_mem = cuda.to_device(np.array([length]))
		k_list_global_mem      = cuda.to_device(np.array([self.k]))

		tpb = (TPB, TPB)
		bpg = (math.ceil(length / TPB), math.ceil(length / TPB))

		snn_cuda[bpg, tpb](
			raw_knn_global_mem, snn_result_global_mem, 
			k_list_global_mem, length_list_global_mem
		)

		snn_result = snn_result_global_mem.copy_to_host()
		return snn_result


