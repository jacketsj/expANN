#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "basic_bench.h"
#include "bench_data_manager.h"

#include "dataset.h"
#include "dataset_loader.h"
#include "ehnsw_engine_basic_fast.h"
#include "ehnsw_engine_basic_fast_multilist.h"
#include "ensg_engine.h"
#include "hnsw_engine_reference.h"

template <typename Engine, typename EngineConfig> struct job {
	EngineConfig conf;
	std::atomic<size_t> claimed;
	std::variant<bench_data, std::string> result;
	job(const job& oth)
			: conf(oth.conf), claimed(static_cast<size_t>(oth.claimed)),
				result(oth.result) {
		if (claimed != 0)
			std::cerr << "Error: Copied claimed job" << std::endl;
	}
	job(const EngineConfig& _conf)
			: conf(_conf), claimed(0), result("Job not yet completed") {}
	template <typename BenchType>
	void run(const BenchType& basic_benchmarker, size_t thread_id) {
		if (claimed++ == 0) {
			Engine eng(conf);

			std::string meta;
			meta += eng.name();
			meta += "(";
			for (const auto& [k, v] : eng.param_list()) {
				meta += k + '=' + v + ',';
			}
			meta[meta.size() - 1] = ')';

			std::printf("Running job (tid=%zu): %s\n", thread_id, meta.c_str());
			result = basic_benchmarker.get_benchmark_data(eng);

			std::string res_str = std::holds_alternative<bench_data>(result)
																? std::get<bench_data>(result).to_string()
																: std::get<std::string>(result);

			std::printf("Completed job (tid=%zu): %s\nResult:%s\n", thread_id,
									meta.c_str(), res_str.c_str());
		}
	}
};

template <typename BenchType, typename JobT>
void perform_benchmarks_on_thread(const BenchType& basic_benchmarker,
																	size_t thread_id, std::vector<JobT>& jobs) {
	for (auto& job : jobs)
		job.run(basic_benchmarker, thread_id);
}

template <typename BenchType, typename JobT, typename... JobArgs>
void perform_benchmarks_on_thread(const BenchType& basic_benchmarker,
																	size_t thread_id, std::vector<JobT>& jobs,
																	JobArgs&... args) {
	perform_benchmarks_on_thread(basic_benchmarker, thread_id, jobs);
	perform_benchmarks_on_thread(basic_benchmarker, thread_id, args...);
}

template <typename BenchType, typename... JobArgs>
void perform_benchmarks_with_threads(const BenchType& basic_benchmarker,
																		 size_t num_threads, JobArgs&... args) {
	std::vector<std::jthread> threads;
	for (size_t t = 0; t < num_threads; ++t) {
		threads.emplace_back([&basic_benchmarker, t, &args...] {
			perform_benchmarks_on_thread(basic_benchmarker, t, args...);
		});
	}
}

template <typename JobT>
void store_benchmark_results(bench_data_manager& bdm,
														 const std::vector<JobT>& jobs) {
	for (const auto& job : jobs)
		bdm.add(job.result);
}

template <typename JobT, typename... JobArgs>
void store_benchmark_results(bench_data_manager& bdm,
														 const std::vector<JobT>& jobs,
														 const JobArgs&... args) {
	store_benchmark_results(bdm, jobs);
	store_benchmark_results(bdm, args...);
}

template <typename BenchType, typename... Args>
auto perform_and_store_benchmark_results(std::string dsname, size_t num_threads,
																				 BenchType& basic_benchmarker,
																				 Args&... args) {
	perform_benchmarks_with_threads(basic_benchmarker, num_threads, args...);
	bench_data_manager bdm(dsname);
	store_benchmark_results(bdm, args...);
	return bdm;
}

template <typename test_dataset_t>
bench_data_manager perform_benchmarks(test_dataset_t ds, size_t num_threads) {
	basic_bench<float, test_dataset_t> basic_benchmarker(ds);

	std::vector<
			job<ehnsw_engine_basic_fast<float>, ehnsw_engine_basic_fast_config>>
			ehnsw_engine_basic_fast_jobs;
	std::vector<job<ehnsw_engine_basic_fast_multilist<float>,
									ehnsw_engine_basic_fast_multilist_config>>
			ehnsw_engine_basic_fast_multilist_jobs;
	std::vector<job<hnsw_engine_reference<float>, hnsw_engine_reference_config>>
			hnsw_engine_reference_jobs;
	std::vector<job<ensg_engine<float>, ensg_engine_config>> ensg_engine_jobs;

	for (size_t k = 60; k <= 100; k += 20) {
		for (size_t num_for_1nn = 3; num_for_1nn <= 6; num_for_1nn += 3) {
			if (true) {
				ensg_engine_jobs.emplace_back(ensg_engine_config(k, num_for_1nn, 1.0f));
			}
			for (size_t edge_count_search_factor : {2, 3}) {
				if (true) {
					ehnsw_engine_basic_fast_jobs.emplace_back(
							ehnsw_engine_basic_fast_config(k, 2 * k, num_for_1nn,
																						 k * edge_count_search_factor));
				}
				if (true) {
					ehnsw_engine_basic_fast_multilist_jobs.emplace_back(
							ehnsw_engine_basic_fast_multilist_config(
									k, 2 * k, num_for_1nn, k * edge_count_search_factor));
				}
				if (true) {
					for (bool use_ecuts : {false}) {
						hnsw_engine_reference_jobs.emplace_back(
								hnsw_engine_reference_config(k, edge_count_search_factor * k,
																						 num_for_1nn, use_ecuts));
					}
				}
			}
		}
	}

	return perform_and_store_benchmark_results(
			ds.name, num_threads, basic_benchmarker, ehnsw_engine_basic_fast_jobs,
			ehnsw_engine_basic_fast_multilist_jobs, hnsw_engine_reference_jobs,
			ensg_engine_jobs);
}
