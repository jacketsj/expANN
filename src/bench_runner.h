#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "basic_bench.h"
#include "bench_data_manager.h"

#include "arrangement_engine.h"
#include "brute_force_engine.h"
#include "dataset.h"
#include "dataset_loader.h"
#include "ehnsw_engine.h"
#include "ehnsw_engine_2.h"
#include "hier_arrangement_engine.h"
#include "hnsw_engine.h"
#include "hnsw_engine_2.h"
#include "hnsw_engine_hybrid.h"
#include "tree_arrangement_engine.h"
#include "tree_arrangement_engine_if.h"

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

template <typename test_dataset_t>
bench_data_manager perform_benchmarks(test_dataset_t ds, size_t num_threads) {
	basic_bench<float, test_dataset_t> basic_benchmarker(ds);

	std::vector<job<hnsw_engine<float, false>, hnsw_engine_config>>
			hnsw_engine_jobs;
	std::vector<job<hnsw_engine_2<float>, hnsw_engine_2_config>>
			hnsw_engine_2_jobs;
	std::vector<job<arrangement_engine<float>, arrangement_engine_config>>
			arrangement_engine_jobs;
	std::vector<job<ehnsw_engine<float>, ehnsw_engine_config>> ehnsw_engine_jobs;
	std::vector<job<ehnsw_engine_2<float>, ehnsw_engine_2_config>>
			ehnsw_engine_2_jobs;
	std::vector<
			job<hier_arrangement_engine<float>, hier_arrangement_engine_config>>
			hier_arrangement_engine_jobs;
	std::vector<job<hnsw_engine_hybrid<float>, hnsw_engine_hybrid_config>>
			hnsw_engine_hybrid_jobs;
	std::vector<
			job<tree_arrangement_engine<float>, tree_arrangement_engine_config>>
			tree_arrangement_engine_jobs;
	std::vector<
			job<tree_arrangement_engine_if<float>, tree_arrangement_engine_if_config>>
			tree_arrangement_engine_if_jobs;

	// TODO implement a way to disable the timeout in get_benchmark_data (e.g. if
	// it's 0, or at least a way to make it super long)
	std::cerr << "Running benchmarks with a 6000s timeout" << std::endl;

	using namespace std::chrono_literals;
	// auto default_timeout = 6000s;

	if (false) {
		// std::cerr << "Running brute force engine." << std::endl;
		// brute_force_engine<float> engine_bf;
		// bdm.add(basic_benchmarker.get_benchmark_data(engine_bf));
		// std::cerr << "Completed brute force engine." << std::endl;
	}

	if (false) {
		for (size_t k = 100; k <= 140; k += 40) {
			for (int p2 = 15; p2 < 18; ++p2) {
				if (p2 > 4)
					p2 += 2;
				// std::cerr << "About to start hnsw(k=" << k << ",p2=" << p2 << ")"
				//					<< std::endl;
				hnsw_engine_jobs.emplace_back(hnsw_engine_config(50, k, 0.5 * p2));
				//  hnsw_engine<float, false> engine(hnsw_engine_config(50, k, 0.5 *
				//  p2)); bdm.add(basic_benchmarker.get_benchmark_data(engine));
				//  std::cerr << "Completed hnsw(k=" << k << ",p2=" << p2 << ")"
				//					<< std::endl;
			}
		}
		// for (size_t k = 2; k <= 128; k += 12) {
		// 	for (size_t num_for_1nn = 32; num_for_1nn <= 64 * 4 * 2; num_for_1nn *=
		// 4)
		// {
	}
	// for (size_t rk = 8; rk * rk <= 140; rk += 8)
	// for (size_t k = rk * rk; k <= 140; k = rk * rk) { // k += 15) {
	// for (size_t k = 100; k <= 100; k += 8) {
	//	for (size_t num_for_1nn = 4; num_for_1nn <= 32; num_for_1nn *= 2) {
	// for (size_t repeats = 0; repeats < 8; ++repeats) {
	// for (size_t k = 28; k <= 28; k += 8) {
	// for (size_t k : {28, 56}) {
	if (true) {
		for (size_t k : {28, 50}) {
			// for (size_t num_for_1nn = 128; num_for_1nn <= 128; num_for_1nn *= 2) {
			for (size_t num_for_1nn = 2; num_for_1nn <= 8; num_for_1nn *= 2) {
				// for (size_t num_for_1nn = 2; num_for_1nn <= 128; num_for_1nn *= 2) {
				//  std::cerr << "About to start hnsw2(k=" << k << ",n4nn=" <<
				//  num_for_1nn
				//					<< ")" << std::endl;
				hnsw_engine_2_jobs.emplace_back(
						hnsw_engine_2_config(100, k, num_for_1nn, true));
				// hnsw_engine_2<float> engine2(hnsw_engine_2_config(100, k,
				// num_for_1nn));
				// bdm.add(basic_benchmarker.get_benchmark_data(engine2));
				//  auto bd = basic_benchmarker.get_benchmark_data(engine2,
				//  default_timeout); std::string res_str =
				//  std::holds_alternative<bench_data>(bd) 													?
				//  std::get<bench_data>(bd).to_string() 													:
				//  std::get<std::string>(bd);
				//  bdm.add(bd);
				// std::cerr << "Completed hnsw2(k=" << k << ")" << std::endl;
				//  std::cerr << "Completed hnsw2(k=" << k << ") bd=" << res_str <<
				//  std::endl;
			}
		}
	}
	//}
	for (size_t ecm = 10; ecm <= 160; ecm *= 2)
		for (size_t mpc = 1; mpc <= 8; mpc *= 2)
			for (size_t nc = 1; nc <= 8; nc *= 2)
				for (size_t n4nn = 1; n4nn <= 16; n4nn *= 4) {
					// for (size_t ecm = 2; ecm <= 10; ecm += 1)
					//	for (size_t mpc = 1; mpc <= 4; mpc *= 2)
					//		for (size_t nc = 1; nc * mpc < ecm; nc *= 2)
					//			for (size_t n4nn = 1; n4nn <= 16; n4nn *= 4)
					// to_run.emplace_back(ecm, 100, mpc, nc, n4nn);
				}
	// to_run.emplace_back(56, 100, 1, 16, 32);
	// to_run.emplace_back(47, 100, 1, 16, 64);
	// to_run.emplace_back(46, 100, 1, 4, 128);

	// for (size_t K = 4; K <= 64; K += 4) {
	//	for (size_t k = 4; k * K <= 128 * 8; k += 4) {
	//		for (size_t num_for_1nn = 32; num_for_1nn <= 64 * 4 * 2;
	// for (size_t K = 4; K <= 256; K *= 2) {
	//	for (size_t k = 16; k <= 128 * 2; k *= 4) {
	//		for (size_t num_for_1nn = 32 * 4; num_for_1nn <= 64 * 2;
	if (true) {
		// for (size_t K = 2; K <= 32; K *= 2) {
		//	for (size_t k = 11; k < 64; k += 9) {
		//		for (size_t num_for_1nn = 4; num_for_1nn <= 64; num_for_1nn *= 2) {
		//			for (size_t min_per_cut = 1;
		//					 min_per_cut * K <= k && min_per_cut <= 16; min_per_cut *= 2) {
		for (size_t k : {28, 50}) {
			// for (size_t k : {28}) {
			for (size_t num_for_1nn = 2; num_for_1nn <= 8; num_for_1nn *= 2) {
				// for (size_t num_for_1nn = 64; num_for_1nn <= 128; num_for_1nn *= 2) {
				// for (size_t num_for_1nn = 2; num_for_1nn <= 2; num_for_1nn *= 2) {
				//  for (size_t K : {2, 4}) {
				for (size_t K : {2}) {
					// for (size_t min_per_cut : {1, 2}) {
					for (size_t min_per_cut : {1}) {
						// std::cerr << "About to start ehnsw2(k=" << k << ",K=" << K
						//					<< ",n4nn=" << num_for_1nn
						//					<< ",min_per_cut=" << min_per_cut << ")" << std::endl;
						ehnsw_engine_2_jobs.emplace_back(ehnsw_engine_2_config(
								100, k, num_for_1nn, K, min_per_cut, true));
						// ehnsw_engine_2<float> engine(
						//		ehnsw_engine_2_config(100, k, num_for_1nn, K, min_per_cut));
						// bdm.add(basic_benchmarker.get_benchmark_data(engine));
						// std::cerr << "Completed ehnsw2(k=" << k << ",K=" << K
						//					<< ",n4nn=" << num_for_1nn
						//					<< ",min_per_cut=" << min_per_cut << ")" << std::endl;
					}
				}
			}
		}
	}
	if (false) {
		for (size_t K = 4; K <= 32; K += 8) {
			for (size_t k = 2; k * K <= 64; k += 12) {
				for (size_t num_for_1nn = 1; num_for_1nn <= 40; num_for_1nn *= 4) {
					// std::cerr << "About to start ehnsw(k=" << k << ",K=" << K
					//					<< ",n4nn=" << num_for_1nn << ")" << std::endl;
					ehnsw_engine_jobs.emplace_back(
							ehnsw_engine_config(100, k, K, num_for_1nn, true));
					// ehnsw_engine<float> engine(
					//		ehnsw_engine_config(100, k, K, num_for_1nn));
					// bdm.add(basic_benchmarker.get_benchmark_data(engine));
					// std::cerr << "Completed ehnsw(k=" << k << ",K=" << K
					//					<< ",n4nn=" << num_for_1nn << ")" << std::endl;
				}
			}
		}
	}
	if (false) {
		for (size_t k = 2; k <= 60; k += 12) {
			for (size_t num_for_1nn = 1; num_for_1nn <= 40; num_for_1nn *= 4) {
				// std::cerr << "About to start hnsw_hybrid(k=" << k
				//					<< ",n4nn=" << num_for_1nn << ")" << std::endl;
				hnsw_engine_hybrid_jobs.emplace_back(
						hnsw_engine_hybrid_config(100, k, num_for_1nn, true));
				// hnsw_engine_hybrid<float> engine(
				//		hnsw_engine_hybrid_config(100, k, num_for_1nn));
				// bdm.add(basic_benchmarker.get_benchmark_data(engine));
				// std::cerr << "Completed hnsw_hybrid(k=" << k << ",n4nn=" <<
				// num_for_1nn
				//					<< ")" << std::endl;
			}
		}

		if (false) {
			for (size_t k = 1; k < 4; ++k) {
				for (size_t n = 4; n <= 256; n *= 2) {
					for (size_t m = 1; m <= 64; m *= 4) {
						// std::cerr << "Starting arrangement(k=" << k << ",n=" << n
						//					<< ",m=" << m << ")" << std::endl;
						arrangement_engine_jobs.emplace_back(
								arrangement_engine_config(k, n, m));
						// arrangement_engine<float> engine(
						//		arrangement_engine_config(k, n, m));
						// bdm.add(basic_benchmarker.get_benchmark_data(engine));
						// std::cerr << "Completed arrangement(k=" << k << ",n=" << n
						//					<< ",m=" << m << ")" << std::endl;
					}
				}

				for (size_t na = 1; na <= 16; na *= 4) {
					for (size_t levels = 1; levels * na <= 32; levels *= 2) {
						for (size_t sc = 32; sc <= 8192; sc *= 4) {
							// std::cerr << "Starting hier arrangement(na=" << na
							//					<< ",levels=" << levels << ",sc=" << sc << ")"
							//					<< std::endl;
							hier_arrangement_engine_jobs.emplace_back(
									hier_arrangement_engine_config(na, levels, sc));
							// hier_arrangement_engine<float> engine(
							//		hier_arrangement_engine_config(na, levels, sc));
							// bdm.add(basic_benchmarker.get_benchmark_data(engine));
							// std::cerr << "Completed hier arrangement(na=" << na
							//					<< ",levels=" << levels << ",sc=" << sc << ")"
							//					<< std::endl;
						}
					}
				}
			}
		}
	}
	if (false) {
		// for (size_t tc = 2; tc <= 64; tc *= 2) {
		for (size_t tc = 32; tc <= 48; tc += 8) {
			// for (size_t max_leaf_size = 64; max_leaf_size <= 1024 * 4;
			// max_leaf_size *= 4) {
			for (size_t max_leaf_size = 1024; max_leaf_size <= 1024 * 4 * 4;
					 max_leaf_size *= 16) {
				for (size_t sc = max_leaf_size / tc; sc * tc <= 8192 * 1; sc *= 16) {
					// for (size_t sc = max_leaf_size; sc * tc <= 8192 * 8; sc *= 16 * 2)
					// {
					//  std::cerr << "Starting tree arrangement(tc=" << tc
					//					<< ",max_leaf_size=" << max_leaf_size << ",sc=" << sc <<
					//")"
					//					<< std::endl;
					//  std::cerr << "Expected time proportional to: " << sc * tc
					//					<< std::endl;
					//  auto begin = std::chrono::high_resolution_clock::now();
					tree_arrangement_engine_jobs.emplace_back(
							tree_arrangement_engine_config(tc, max_leaf_size, sc));
					// tree_arrangement_engine<float> engine(
					//		tree_arrangement_engine_config(tc, max_leaf_size, sc));
					// bdm.add(basic_benchmarker.get_benchmark_data(engine));
					// auto end = std::chrono::high_resolution_clock::now();
					// std::cerr << "Actual time: "
					//					<< std::chrono::duration_cast<std::chrono::nanoseconds>(
					//								 end - begin)
					//								 .count()
					//					<< "ns" << std::endl;
					// std::cerr << "Completed tree arrangement(tc=" << tc
					//					<< ",max_leaf_size=" << max_leaf_size << ",sc=" << sc <<
					//")"
					//					<< std::endl;
				}
			}
		}
	}
	if (false) {
		for (size_t tc = 2; tc <= 64; tc *= 2) {
			for (size_t max_leaf_size = 64; max_leaf_size <= 1024 * 4;
					 max_leaf_size *= 4) {
				for (size_t sc = max_leaf_size; sc * tc <= 8192 * 8; sc *= 16 * 2) {
					// std::cerr << "Starting tree arrangement_if(tc=" << tc
					//					<< ",max_leaf_size=" << max_leaf_size << ",sc=" << sc <<
					//")"
					//					<< std::endl;
					// std::cerr << "Expected time proportional to: " << sc * tc
					//					<< std::endl;
					// auto begin = std::chrono::high_resolution_clock::now();
					tree_arrangement_engine_if_jobs.emplace_back(
							tree_arrangement_engine_if_config(tc, max_leaf_size, sc));
					// tree_arrangement_engine_if<float> engine(
					//		tree_arrangement_engine_if_config(tc, max_leaf_size, sc));
					// bdm.add(basic_benchmarker.get_benchmark_data(engine));
					// auto end = std::chrono::high_resolution_clock::now();
					// std::cerr << "Actual time: "
					//					<< std::chrono::duration_cast<std::chrono::nanoseconds>(
					//								 end - begin)
					//								 .count()
					//					<< "ns" << std::endl;
					// std::cerr << "Completed tree arrangement_if(tc=" << tc
					//					<< ",max_leaf_size=" << max_leaf_size << ",sc=" << sc <<
					//")"
					//					<< std::endl;
				}
			}
		}
	}
	perform_benchmarks_with_threads(
			basic_benchmarker, num_threads, hnsw_engine_jobs, hnsw_engine_2_jobs,
			arrangement_engine_jobs, ehnsw_engine_jobs, ehnsw_engine_2_jobs,
			hier_arrangement_engine_jobs, hnsw_engine_hybrid_jobs,
			tree_arrangement_engine_jobs, tree_arrangement_engine_if_jobs);

	bench_data_manager bdm(ds.name);
	store_benchmark_results(bdm, hnsw_engine_jobs, hnsw_engine_2_jobs,
													arrangement_engine_jobs, ehnsw_engine_jobs,
													ehnsw_engine_2_jobs, hier_arrangement_engine_jobs,
													hnsw_engine_hybrid_jobs, tree_arrangement_engine_jobs,
													tree_arrangement_engine_if_jobs);

	return bdm;
}
