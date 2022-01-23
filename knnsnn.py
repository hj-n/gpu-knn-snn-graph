from numba import cuda

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



'''
INPUT
- data: 2D array (N, D) where N denotes the number of points and D denotes the dimensionality
- k: k value used to compute initial knn and snn

'''
class KnnSnn:
	
	def __init__(self, k=20):
		self.k=k

	
	def knn():
		pass

	def snn(matrix, ):
		pass