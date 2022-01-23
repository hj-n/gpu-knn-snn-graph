# GPU implementation of *k*NN and SNN 

- GPU implementation of $k$-Nearest Neighbors and Shared-Nearest Neighbors 
- Supported by `numba cuda` and `faiss` library

### Env Initialization & Activation

Prior to the import and execution of main source code (`knnsnn.py`), a conda envrionment should be set. Execute following commands to set the envrionment.

```sh
conda env create --file ksnn_env.yaml
conda activate gpu-knn-snn
```

### Import & Execution

Place `knnsnn.py` in the working directory, and import the within class using

```python
from knnsnn import KnnSnn as ks
```

Afterwards, you can create an instance and runn knn and snn by 

```python
KSnn = ks(k)
knn_indices = KSnn.knn(sample_data)
snn_results = KSnn.snn(knn_indices)
```

Refer to `test.py` to know the way to use `knnsnn.py` in detail.
