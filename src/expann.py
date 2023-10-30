import numpy as np
import expann_py

from base import BaseOODANN

print(expann_py)

class ExpAnnWrapper(BaseOODANN):
    def __init__(self):
        self.engine = expann_py.expANN()
        self.name = self.engine.name()
        self.res = None

    def fit(self, dataset):
        for vector in dataset:
            v = expann_py.Vec(vector.tolist())
            self.engine.store_vector(v)
        self.engine.build()

    def query(self, X, k):
        query_vectors = []
        for query_vector in X:
            v = expann_py.Vec(query_vector.tolist())
            query_vectors.append(v)
        result_indices = self.engine.query_k_batch(query_vectors, k)
        self.res = np.array(result_indices)

    def get_results(self):
        if self.res is None:
            raise ValueError("Run a query before getting results")
        return self.res

    def __str__(self):
        return self.name
