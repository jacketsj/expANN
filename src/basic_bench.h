#pragma once

#include <chrono>
#include <future>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "ann_engine.h"
#include "bench_data.h"
#include "randomgeometry.h"
#include "vec.h"

const double TOLERANCE = 1e-7;

template <typename T> struct basic_bench {
	std::vector<vec<T>> dataset;
	std::vector<vec<T>> query_vecs;
	std::vector<vec<T>> query_ans;
	basic_bench(size_t n, size_t m);
	void gen_dataset(size_t n, size_t m);
	template <class Engine> void populate_ans(ann_engine<T, Engine>& eng) {
		query_ans.clear(); // in case of repeat calls to this function

		// assume eng is an optimal solver

		for (const auto& v : dataset)
			eng.store_vector(v);
		eng.build();

		for (const auto& q : query_vecs) {
			const auto& ans = eng.query(q);
			query_ans.push_back(ans);
		}
	}
	template <class Engine, class Rep, class Period>
	std::variant<bench_data, std::string> get_benchmark_data(
			ann_engine<T, Engine>& eng,
			const std::chrono::duration<Rep, Period>& timeout_duration) const {
		std::stop_source ssource;
		std::packaged_task<bench_data()> task([&]() {
			return get_benchmark_data_no_timeout(eng, ssource.get_token());
		});
		auto future = task.get_future();
		auto time_run_begin = std::chrono::high_resolution_clock::now();
		std::thread t(std::move(task));
		if (future.wait_for(timeout_duration) != std::future_status::timeout) {
			t.join();
			return future.get();
		} else {
			ssource.request_stop();
			t.join();
			auto time_run_end = std::chrono::high_resolution_clock::now();
			return "Task timed out after " +
						 std::to_string(
								 std::chrono::duration<double>(
										 std::chrono::duration_cast<std::chrono::nanoseconds>(
												 time_run_end - time_run_begin))
										 .count()) +
						 "/" +
						 std::to_string(
								 std::chrono::duration<double>(timeout_duration).count()) +
						 " seconds.";
		}
	}
	template <class Engine>
	bench_data get_benchmark_data_no_timeout(ann_engine<T, Engine>& eng,
																					 std::stop_token stoken) const {
		bench_data ret;

		// record the store and build timespan
		auto time_begin_build = std::chrono::high_resolution_clock::now();
		// store all vectors in the engine
		for (const auto& v : dataset)
			eng.store_vector(v);
		if (stoken.stop_requested())
			return ret;
		// build the engine
		eng.build();
		auto time_end_build = std::chrono::high_resolution_clock::now();

		if (stoken.stop_requested())
			return ret;

		// run the queries
		double avg_dist = 0, avg_dist2 = 0;
		size_t num_best_found = 0;
		size_t i = 0;
		auto time_begin = std::chrono::high_resolution_clock::now();
		for (const auto& q : query_vecs) {
			const auto& ans = eng.query(q);
			T d = dist(q, ans), d2 = dist2(q, ans);
			avg_dist += d;
			avg_dist2 += d2;
			if (!query_ans.empty() && d <= dist(q, query_ans[i++]) + TOLERANCE)
				++num_best_found;

			if (stoken.stop_requested())
				return ret;
		}
		auto time_end = std::chrono::high_resolution_clock::now();

		ret.time_per_query_ns =
				double(std::chrono::duration_cast<std::chrono::nanoseconds>(time_end -
																																		time_begin)
									 .count()) /
				query_vecs.size();
		ret.time_to_build_ns =
				double(std::chrono::duration_cast<std::chrono::nanoseconds>(
									 time_end_build - time_begin_build)
									 .count());
		ret.average_distance = double(avg_dist) / query_vecs.size();
		ret.average_squared_distance = double(avg_dist2) / query_vecs.size();
		ret.recall = double(num_best_found) / query_vecs.size();

		ret.engine_name = eng.name();

		return ret;
	}
	template <class Engine>
	void perform_benchmark(ann_engine<T, Engine>& eng,
												 bool print_basics = false) const {
		// store all vectors in the engine
		for (const auto& v : dataset)
			eng.store_vector(v);
		// build the engine
		eng.build();
		// run the queries
		double avg_dist = 0, avg_dist2 = 0;
		auto time_begin = std::chrono::high_resolution_clock::now();
		for (const auto& q : query_vecs) {
			const auto& ans = eng.query(q);
			T d = dist(q, ans), d2 = dist2(q, ans);
			avg_dist += d;
			avg_dist2 += d2;
		}
		auto time_end = std::chrono::high_resolution_clock::now();
		avg_dist /= query_vecs.size();
		avg_dist2 /= query_vecs.size();
		auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
													 time_end - time_begin)
													 .count();
		if (print_basics) {
			std::cout << avg_dist << "\t" << avg_dist2 << "\t" << duration_ns;
		} else {
			std::cout << "Benchmarking " << eng.name() << '\n';
			std::cout << "\taverage distance: " << avg_dist << '\n';
			std::cout << "\taverage squared distance: " << avg_dist2 << '\n';
			std::cout << "\taverage query time: " << duration_ns / query_vecs.size()
								<< "ns" << std::endl;
		}
	}
};

template <typename T> void basic_bench<T>::gen_dataset(size_t n, size_t m) {
	vec_generator<T> vg;
	for (size_t i = 0; i < n; ++i) {
		dataset.push_back(vg.random_vec());
	}
	for (size_t i = 0; i < m; ++i) {
		query_vecs.push_back(vg.random_vec());
	}
}

template <typename T> basic_bench<T>::basic_bench(size_t n, size_t m) {
	gen_dataset(n, m);
}
