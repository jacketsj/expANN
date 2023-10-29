import numpy as np
from expann import ExpAnnWrapper
from scipy.spatial import distance

def brute_force_knn(data, queries, k):
    dists = distance.cdist(queries, data, 'euclidean')
    nn_indices = np.argsort(dists, axis=1)[:, :k]
    return nn_indices

def compute_recall(predicted, truth):
    total_recall = 0.0
    for p, t in zip(predicted, truth):
        total_recall += len(set(p) & set(t)) / len(t)
    return total_recall / len(predicted)

# Initialize ANN wrapper
ann = ExpAnnWrapper()
# Generate random dataset with vectors of length 128
dataset = np.random.rand(10000, 128).astype(np.float32)
# Build the index
ann.fit(dataset)
# Generate random query points with vectors of length 128
query_points = np.random.rand(5, 128).astype(np.float32)
# Perform k-NN query
k = 5
ann.query(query_points, k)
# Get results from ANN
ann_results = ann.get_results()
print("ANN Results:\n", ann_results)

# Compute ground truth using brute-force
ground_truth = brute_force_knn(dataset, query_points, k)
print("Ground Truth:\n", ground_truth)

# Compute recall
recall = compute_recall(ann_results, ground_truth)
print(f"Recall: {recall:.4f}")
