#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "basic_bench.h"
#include "bench_data_manager.h"

#include "antitopo_engine.h"
#include "dataset.h"
#include "dataset_loader.h"
#include "hnsw_engine_reference.h"
#include "line_pruning_exact_engine.h"
#include "par_antitopo_engine.h"

template <typename Engine, typename EngineConfig> struct job {
	EngineConfig conf;
	std::atomic<size_t> claimed;
	std::variant<bench_data, std::string> result;
	size_t job_index;
	job(const job& oth)
			: conf(oth.conf), claimed(static_cast<size_t>(oth.claimed)),
				result(oth.result), job_index(oth.job_index) {
		if (claimed != 0)
			std::cerr << "Error: Copied claimed job" << std::endl;
	}
	job(const EngineConfig& _conf, size_t _job_index)
			: conf(_conf), claimed(0), result("Job not yet completed"),
				job_index(_job_index) {}
	template <typename BenchType>
	void run(const BenchType& basic_benchmarker, size_t thread_id,
					 size_t total_jobs) {
		if (claimed++ == 0) {
			Engine eng(conf);

			std::string meta;
			meta += eng.name();
			meta += "(";
			for (const auto& [k, v] : eng.param_list()) {
				meta += k + '=' + v + ',';
			}
			meta[meta.size() - 1] = ')';

			std::printf("Running job %zu/%zu (tid=%zu): %s\n", job_index + 1,
									total_jobs, thread_id, meta.c_str());
			fflush(stdout);
			result = basic_benchmarker.get_benchmark_data(eng);

			std::string res_str = std::holds_alternative<bench_data>(result)
																? std::get<bench_data>(result).to_string()
																: std::get<std::string>(result);

			std::printf("Completed job %zu/%zu (tid=%zu): %s\nResult:%s\n",
									job_index + 1, total_jobs, thread_id, meta.c_str(),
									res_str.c_str());
			fflush(stdout);
		}
	}
};

template <typename BenchType, typename JobT>
void perform_benchmarks_on_thread(const BenchType& basic_benchmarker,
																	size_t thread_id, size_t total_jobs,
																	std::vector<JobT>& jobs) {
	for (auto& job : jobs)
		job.run(basic_benchmarker, thread_id, total_jobs);
}

template <typename BenchType, typename JobT, typename... JobArgs>
void perform_benchmarks_on_thread(const BenchType& basic_benchmarker,
																	size_t thread_id, size_t total_jobs,
																	std::vector<JobT>& jobs, JobArgs&... args) {
	perform_benchmarks_on_thread(basic_benchmarker, thread_id, total_jobs, jobs);
	perform_benchmarks_on_thread(basic_benchmarker, thread_id, total_jobs,
															 args...);
}

template <typename BenchType, typename... JobArgs>
void perform_benchmarks_with_threads(const BenchType& basic_benchmarker,
																		 size_t num_threads, size_t total_jobs,
																		 JobArgs&... args) {
	std::vector<std::jthread> threads;
	for (size_t t = 0; t < num_threads; ++t) {
		threads.emplace_back([&basic_benchmarker, t, total_jobs, &args...] {
			perform_benchmarks_on_thread(basic_benchmarker, t, total_jobs, args...);
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
																				 size_t total_jobs,
																				 BenchType& basic_benchmarker,
																				 Args&... args) {
	perform_benchmarks_with_threads(basic_benchmarker, num_threads, total_jobs,
																	args...);
	bench_data_manager bdm(dsname);
	store_benchmark_results(bdm, args...);
	return bdm;
}

template <typename Engine> using JobType = job<Engine, typename Engine::config>;
template <typename... Engines>
using JobTuple = std::tuple<std::vector<JobType<Engines>>...>;

#define ADD_JOB(Engine, ...)                                                   \
	std::get<std::vector<JobType<Engine>>>(job_lists).emplace_back(              \
			Engine::config(__VA_ARGS__), job_index++)

template <typename test_dataset_t>
bench_data_manager perform_benchmarks(test_dataset_t ds, size_t num_threads) {

	basic_bench<float, test_dataset_t> basic_benchmarker(ds);

	size_t job_index = 0;

	JobTuple<hnsw_engine_reference<float>, antitopo_engine<float>,
					 line_pruning_exact_engine<float>, par_antitopo_engine>
			job_lists;

	for (size_t k = 30; k <= 60; k += 15) {
		for (size_t num_for_1nn : {1, 2, 3, 4, 5}) {					// 5
			for (size_t edge_count_search_factor : {500 / k}) { // 3
				for (bool use_compression : {false}) {
					for (size_t build_threads : {14}) {
						for (bool use_mips : {false}) {
							if (false) {
								ADD_JOB(par_antitopo_engine, k, k * edge_count_search_factor,
												build_threads, num_for_1nn, use_mips);
							}
						}
					}
					for (bool use_largest_direction_filtering : {false}) {
						for (size_t ortho_count : {1, 3}) { // 1,3,5
							for (float ortho_factor :
									 (ortho_count == 1
												? std::vector({0.5f})
												: std::vector(
															{1.0f, 0.5f, 0.4f}))) { // 1.0f, 0.5f, 2.0f
								for (float ortho_bias :
										 ortho_count == 1
												 ? std::vector({0.0f})
												 : std::vector({0.0f})) { //,1.0f, 1000000000.0f})) {
									for (size_t prune_overflow : {0, 1}) { // 0,1,3
										if (true) {
											std::string filename = "index/sift";
											filename += "_k" + std::to_string(k);
											filename +=
													"_efx" + std::to_string(edge_count_search_factor);
											filename += "_orthocount" + std::to_string(ortho_count);
											filename += "_orthofactor" + std::to_string(ortho_factor);
											filename += "_orthobias" + std::to_string(ortho_bias);
											filename +=
													"_pruneoverflow" + std::to_string(prune_overflow);
											ADD_JOB(antitopo_engine<float>, k, 2 * k, num_for_1nn,
															k * edge_count_search_factor, ortho_count,
															ortho_factor, ortho_bias, prune_overflow,
															use_compression, use_largest_direction_filtering,
															filename, true, true);
										}
									}
								}
							}
						}
						if (false) {
							// for (size_t brute_force_size : {64, 128})
							// for (size_t line_count : {128}) {
							/*
							ADD_JOB(line_pruning_exact_engine<float>, brute_force_size,
											line_count,
											antitopo_engine_config(
													k, 2 * k, num_for_1nn,
													k * edge_count_search_factor, use_compression,
													use_largest_direction_filtering));
													*/
							//}
						}
					}
				}
				if (false) {
					for (bool use_ecuts : {true, false}) {
						ADD_JOB(hnsw_engine_reference<float>, k,
										edge_count_search_factor * k, num_for_1nn, use_ecuts);
					}
				}
			}
		}
	}

	return std::apply(
			[&](auto&... jobs) {
				return perform_and_store_benchmark_results(
						ds.name, num_threads, job_index, basic_benchmarker, jobs...);
			},
			job_lists);
}
