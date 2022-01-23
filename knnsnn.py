from numba import cuda
import faiss

@cuda.jit
def snn(raw_knn, snn_result, k_list, length_list):
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
	
	def __init__(self, k=20):
		self.k=k


	'''
	INPUT
	- data: 2D numpy array (N, D) where 
	  - N denotes the number of points 
		- D denotes the dimensionality
	'''
	def knn(self, data):

		index = faiss.IndexFlatL2(data.shape[1])
		index.add(data)

		_, indices = index.search(data, self.k + 1)
		return indices

	def snn(self, matrix):
		pass


