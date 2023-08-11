#pragma once

#include <chrono>
#include <future>
#include <set>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "ann_engine.h"
#include "bench_data.h"
#include "dataset.h"
#include "randomgeometry.h"
#include "vec.h"

const double TOLERANCE = 1e-7;

template <typename T, typename test_dataset_t> struct basic_bench {
	const test_dataset_t& ds;
	basic_bench(const test_dataset_t& _ds) : ds(_ds) {}
	template <class Engine, class Rep, class Period>
	std::variant<bench_data, std::string> get_benchmark_data(
			ann_engine<T, Engine>& eng,
			const std::chrono::duration<Rep, Period>& timeout_duration) const {
		std::stop_source ssource;

		// uncommenting the following runs without a separate thread or timeout
		// return get_benchmark_data_no_timeout(eng, ssource.get_token());

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
		for (size_t i = 0; i < ds.n; ++i)
			eng.store_vector(ds.get_vec(i));
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
		auto time_begin = std::chrono::high_resolution_clock::now();
		for (size_t q = 0; q < ds.m; ++q) {
			// TODO make this kNN instead of 1NN
			const auto& ans = eng.query_k(ds.get_query(q), ds.k_want);
			assert(ans.size() <= ds.k_want);
			// computation time for if this is good is assumed to be negligible
			// compared to query_k time in benchmark
			if (!ans.empty()) {
				T d = dist(ds.get_query(q), ds.get_vec(ans[0])),
					d2 = dist2(ds.get_query(q), ds.get_vec(ans[0]));
				avg_dist += d;
				avg_dist2 += d2;
			} else {
				avg_dist += 1e8; // distance is very large
			}
			const auto& expected_ans = ds.get_query_ans(q);
			// recall is number of ans that appear in expected ans, divided by k_want
			std::set<size_t> ans_s;
			for (auto& vi : ans)
				ans_s.emplace(vi);
			// num_best_found += ans_s.size();
			for (auto& vi : expected_ans)
				if (ans_s.contains(vi)) {
					++num_best_found;
				}

			if (stoken.stop_requested())
				return ret;
		}
		auto time_end = std::chrono::high_resolution_clock::now();

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
		std::cout << "num_best_found=" << num_best_found << "/" << ds.m * ds.k_want
							<< std::endl;
		ret.recall = double(num_best_found) / double(ds.m * ds.k_want);
		ret.param_list = eng.param_list();

		ret.engine_name = eng.name();

		return ret;
	}
};
