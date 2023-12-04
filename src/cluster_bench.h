#pragma once

#include <chrono>
#include <future>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "ann_engine.h"
#include "cluster_bench_data.h"
#include "dataset.h"
#include "randomgeometry.h"
#include "vec.h"

#include <valgrind/callgrind.h>

template <typename T, typename test_dataset_t> struct cluster_bench {
	const test_dataset_t& ds;
	cluster_bench(const test_dataset_t& _ds) : ds(_ds) {}
	template <class Engine>
	/*
	cluster_bench_data get_benchmark_data(ann_engine<T, Engine>& eng) const {
		cluster_bench_data ret;

		// record the store and build timespan
		auto time_begin_build = std::chrono::high_resolution_clock::now();
		// store all vectors in the engine
		for (size_t i = 0; i < ds.n; ++i)
			eng.store_vector(ds.get_vec(i));
		//  build the engine
		eng.build();
		auto time_end_build = std::chrono::high_resolution_clock::now();

		CALLGRIND_START_INSTRUMENTATION;
		CALLGRIND_TOGGLE_COLLECT;

		// run the queries
		double avg_dist = 0, avg_dist2 = 0;
		size_t num_best_found = 0;
		auto time_begin = std::chrono::high_resolution_clock::now();
		for (size_t q = 0; q < ds.m; ++q) {
			std::vector<size_t> ans = eng.query_k(ds.get_query(q), ds.k);
			if (ans.size() > 0) {
				T d = dist(ds.get_query(q), ds.get_vec(ans[0])),
					d2 = dist2(ds.get_query(q), ds.get_vec(ans[0]));
				avg_dist += d;
				avg_dist2 += d2;
			}

			std::set<size_t> ans_s;
			for (auto& i : ans)
				ans_s.emplace(i);
			if (ans_s.size() != ans.size()) {
				std::cerr << "Duplicates detected, engine is buggy." << std::endl;
				assert(ans_s.size() == ans.size());
			}

			std::vector<size_t> expected_ans = ds.get_query_ans(q);
			assert(expected_ans.size() == ds.k);

			size_t intersection_size = 0;
			for (auto& i : expected_ans) {
				if (ans_s.contains(i))
					++intersection_size;
			}
			num_best_found += intersection_size;
		}
		auto time_end = std::chrono::high_resolution_clock::now();

		CALLGRIND_TOGGLE_COLLECT;
		CALLGRIND_STOP_INSTRUMENTATION;

		ret.time_per_query_ns =
				double(std::chrono::duration_cast<std::chrono::nanoseconds>(time_end -
																																		time_begin)
									 .count()) /
				ds.m;
		ret.time_to_build_ns =
				double(std::chrono::duration_cast<std::chrono::nanoseconds>(
									 time_end_build - time_begin_build)
									 .count());
		ret.average_distance = double(avg_dist) / ds.m;
		ret.average_squared_distance = double(avg_dist2) / ds.m;

		ret.recall = double(num_best_found) / (ds.m * ds.k);
		ret.param_list = eng.param_list();

		ret.engine_name = eng.name();

		return ret;
	}
	*/

	cluster_bench_data get_benchmark_data(ann_engine<T, Engine>& eng) const {
		cluster_bench_data ret;
		auto time_begin_build = std::chrono::high_resolution_clock::now();

		std::vector<vec<T>> centroids;
		std::vector<int> assignments(ds.n);
		bool centroids_changed = true;
		std::mt19937 rng; // Random number generator

		// K-means|| initialization
		centroids.push_back(
				ds.get_vec(rng() % ds.n)); // Randomly select the first centroid
		for (int iter = 0; iter < log(ds.n); ++iter) {
			auto eng_copy = eng;
			for (const auto& c : centroids) {
				eng_copy.store_vector(c);
			}
			eng_copy.build();

			std::vector<vec<T>> new_centroids;
			for (int i = 0; i < ds.n; ++i) {
				int nearest = eng_copy.query_k(ds.get_vec(i), 1);
				double dist_squared = dist2(ds.get_vec(i), centroids[nearest]);
				if ((rng() / static_cast<double>(rng.max())) < (dist_squared / ds.n)) {
					new_centroids.push_back(ds.get_vec(i));
				}
			}
			centroids.insert(centroids.end(), new_centroids.begin(),
											 new_centroids.end());
		}

		// unaccelerated K-means++ on the weighted set of centroids to get exactly k
		// centroids
		// TODO do this with an engine
		std::vector<vec<T>> final_centroids;
		final_centroids.push_back(
				centroids[rng() %
									centroids.size()]); // Randomly select the first centroid

		for (int i = 1; i < ds.k; ++i) {
			std::vector<double> cumulative_weights;
			double total_weight = 0.0;

			for (const auto& centroid : centroids) {
				double min_dist_squared = std::numeric_limits<double>::max();
				for (const auto& selected_centroid : final_centroids) {
					double dist_squared = dist2(centroid, selected_centroid);
					if (dist_squared < min_dist_squared) {
						min_dist_squared = dist_squared;
					}
				}
				total_weight += min_dist_squared;
				cumulative_weights.push_back(total_weight);
			}

			double r = std::uniform_real_distribution<>(0.0, total_weight)(rng);
			auto it = std::lower_bound(cumulative_weights.begin(),
																 cumulative_weights.end(), r);
			final_centroids.push_back(
					centroids[std::distance(cumulative_weights.begin(), it)]);
		}

		centroids = std::move(final_centroids);

		while (centroids_changed) {
			centroids_changed = false;
			auto eng_copy = eng;

			// Store centroids in engine and build
			for (const auto& centroid : centroids) {
				eng_copy.store_vector(centroid);
			}
			eng_copy.build();

			// Assign points to nearest centroid
			for (int i = 0; i < ds.n; ++i) {
				int nearest = eng_copy.query_k(ds.get_vec(i), 1);
				if (assignments[i] != nearest) {
					assignments[i] = nearest;
					centroids_changed = true;
				}
			}

			// Update centroids
			std::vector<vec<T>> new_centroids(ds.k, vec<T>::zeroes(ds.d));
			std::vector<int> counts(ds.k, 0);
			for (int i = 0; i < ds.n; ++i) {
				new_centroids[assignments[i]] += ds.get_vec(i);
				counts[assignments[i]]++;
			}
			for (int i = 0; i < ds.k; ++i) {
				if (counts[i] > 0) {
					centroids[i] = new_centroids[i] / static_cast<T>(counts[i]);
				}
			}
		}

		// Calculate score
		ret.score = 0;
		for (int i = 0; i < ds.n; ++i) {
			ret.score += dist2(ds.get_vec(i), centroids[assignments[i]]);
		}

		auto time_end_build = std::chrono::high_resolution_clock::now();
		ret.total_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
														time_end_build - time_begin_build)
														.count();

		return ret;
	}
};
