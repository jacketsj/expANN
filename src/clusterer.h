#pragma once

#include <Eigen/Dense>
#include <random>

#include "vec.h"

class clusterer {
	std::random_device rd;
	std::mt19937 gen;

public:
	clusterer() : rd(), gen(rd()) {}
	void k_means(const std::vector<Eigen::VectorXf>& data, size_t k,
							 std::vector<size_t>& labels,
							 std::vector<Eigen::VectorXf>& centroids) {
		std::uniform_int_distribution<> distrib(0, data.size() - 1);

		size_t dimension = data[0].size();

		// Initialize centroids randomly
		centroids.resize(k);
		for (size_t i = 0; i < k; ++i)
			centroids[i] = data[distrib(gen)];

		labels.resize(data.size());

		bool changed = true;
		while (changed) {
			changed = false;

			// Assign labels
			for (size_t i = 0; i < data.size(); ++i) {
				float minDist = std::numeric_limits<float>::max();
				size_t bestLabel = 0;
				for (size_t j = 0; j < k; ++j) {
					float dist = (data[i] - centroids[j]).squaredNorm();
					if (dist < minDist) {
						minDist = dist;
						bestLabel = j;
					}
				}
				if (labels[i] != bestLabel) {
					labels[i] = bestLabel;
					changed = true;
				}
			}

			// Update centroids
			std::vector<size_t> counts(k, 0);
			std::vector<Eigen::VectorXf> newCentroids(
					k, Eigen::VectorXf::Zero(dimension));
			for (size_t i = 0; i < data.size(); ++i) {
				newCentroids[labels[i]] += data[i];
				counts[labels[i]]++;
			}
			for (size_t j = 0; j < k; ++j)
				if (counts[j] > 0)
					centroids[j] = newCentroids[j] / counts[j];
		}
	}
};
