import numpy as np
import expann_py

class ExpAnnWrapper(BaseANN):
    def __init__(self):
        self.engine = expann_py.AnnEngine()
        self.name = self.engine.name()
        self.res = None

    def fit(self, dataset):
        # Assuming dataset is a 2D numpy array where each row is a vector
        for vector in dataset:
            v = expann_py.Vec(vector.tolist())
            self.engine.store_vector(v)
        self.engine.build()

    def query(self, X, k):
        result_indices = []
        for query_vector in X:
            v = expann_py.Vec(query_vector.tolist())
            indices = self.engine.query_k(v, k)
            result_indices.append(indices)
        self.res = np.array(result_indices)

    def get_results(self):
        if self.res is None:
            raise ValueError("Run a query before getting results")
        return self.res

    def __str__(self):
        return self.name
