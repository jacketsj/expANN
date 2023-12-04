#pragma once

#include <chrono>
#include <future>
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
};
