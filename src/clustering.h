#pragma once

#include <vector>

#include "antitopo_engine.h"
#include "vec.h"

#include <random>

#include "randomgeometry.h"

std::vector<std::vector<size_t>>
accel_k_means(const std::vector<vec<float>::Underlying>& vecs, size_t k,
							antitopo_engine_config conf) {
	// size_t min_cluster_size = 32;
	// size_t max_cluster_size = min_cluster_size * 4;
	std::vector<vec<float>> centroids;
	std::vector<std::vector<size_t>> clusters;
	std::mt19937 gen(0);
	std::uniform_int_distribution<> distribution(0, vecs.size() - 1);
	// std::uniform_real_distribution<> distribution(0, 1);
	for (size_t i = 0; i < k; ++i)
		centroids.emplace_back(vecs[distribution(gen)]);
	/*
	for (size_t i = 0; i < vecs.size(); ++i) {
		if (floor(-log(distribution(gen)) * 1 / log(double(60))) > 0) {
			centroids.emplace_back(vecs[i]);
		}
	}
	*/
	auto assign_to_clusters = [&]() {
		antitopo_engine<float> sub_engine(conf);
		for (auto& v : centroids)
			sub_engine._store_vector(v, true);
		sub_engine.build();
		clusters.clear();
		clusters.resize(centroids.size());
		for (size_t i = 0; i < vecs.size(); ++i) {
			auto sub_engine_results = sub_engine.query_k(vec<float>(vecs[i]), 1);
			clusters[sub_engine_results[0]].emplace_back(i);
		}
	};
	auto compute_centroids = [&]() {
		centroids.resize(clusters.size());
		for (size_t centroid_index = 0; centroid_index < centroids.size();
				 ++centroid_index) {
			auto& centroid = centroids[centroid_index];
			centroid.clear();
			for (size_t elem_index : clusters[centroid_index]) {
				centroid += vec<float>(vecs[elem_index]);
			}
			if (!clusters[centroid_index].empty()) {
				centroid /= clusters[centroid_index].size();
			}
		}
	};
	const size_t max_iters = 30;
	vec_generator<float> rvgen(vec<float>(vecs[0]).size());
	for (size_t iter = 0; iter < max_iters; ++iter) {
		assign_to_clusters();
		/*
		// loosely enforce min/max cluster sizes
		std::vector<std::vector<size_t>> clusters_new;
		for (size_t cluster_index = 0; cluster_index < clusters.size();
				 ++cluster_index) {
			// enforce min cluster size by deleting small clusters
			if (clusters[cluster_index].size() >= min_cluster_size) {
				// enforce max cluster size by splitting big clusters
				if (clusters[cluster_index].size() <= max_cluster_size) {
					clusters_new.emplace_back(clusters[cluster_index]);
				} else {
					// partition with a random hyperplane
					auto project_vec = rvgen.random_vec(); // normal vector of hyperplane
					clusters_new.emplace_back();
					clusters_new.emplace_back();
					for (size_t entry_index : clusters[cluster_index]) {
						clusters_new[clusters_new.size() - 1 -
												 ((vecs[entry_index] - centroids[cluster_index])
															.dot(project_vec) > 0)]
								.emplace_back(entry_index);
					}
				}
			}
		}
		if (!clusters_new.empty()) // don't let all clusters be deleted
			clusters = clusters_new;
		*/
		compute_centroids();
	}
	assign_to_clusters();
	return clusters;
}
