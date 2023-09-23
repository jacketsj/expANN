#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "basic_bench.h"
#include "bench_data_manager.h"

#include "arrangement_engine.h"
#include "brute_force_engine.h"
#include "clustered_ehnsw_engine.h"
#include "dataset.h"
#include "dataset_loader.h"
#include "disk_ehnsw_engine.h"
#include "ehnsw_engine.h"
#include "ehnsw_engine_2.h"
#include "ehnsw_engine_3.h"
#include "filter_ehnsw_engine.h"
#include "hier_arrangement_engine.h"
#include "hnsw_engine.h"
#include "hnsw_engine_2.h"
#include "hnsw_engine_hybrid.h"
#include "hyper_hnsw_engine.h"
#include "jamana_ehnsw_engine.h"
#include "projection_engine.h"
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

template <typename BenchType, typename... Args>
auto& perform_and_store_benchmark_results(std::string dsname,
																					size_t num_threads,
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

	std::vector<job<hnsw_engine<float, false>, hnsw_engine_config>>
			hnsw_engine_jobs;
	std::vector<job<hnsw_engine_2<float>, hnsw_engine_2_config>>
			hnsw_engine_2_jobs;
	std::vector<job<hyper_hnsw_engine<float>, hyper_hnsw_engine_config>>
			hyper_hnsw_engine_jobs;
	std::vector<job<projection_engine<float, hnsw_engine_2<float>>,
									projection_engine_config<hnsw_engine_2_config>>>
			projection_hnsw_engine_2_jobs;
	std::vector<job<arrangement_engine<float>, arrangement_engine_config>>
			arrangement_engine_jobs;
	std::vector<job<ehnsw_engine<float>, ehnsw_engine_config>> ehnsw_engine_jobs;
	std::vector<job<ehnsw_engine_2<float>, ehnsw_engine_2_config>>
			ehnsw_engine_2_jobs;
	std::vector<job<ehnsw_engine_3<float>, ehnsw_engine_3_config>>
			ehnsw_engine_3_jobs;
	std::vector<job<jamana_ehnsw_engine<float>, jamana_ehnsw_engine_config>>
			jamana_ehnsw_engine_jobs;
	std::vector<job<filter_ehnsw_engine<float>, filter_ehnsw_engine_config>>
			filter_ehnsw_engine_jobs;
	std::vector<job<clustered_ehnsw_engine<float>, clustered_ehnsw_engine_config>>
			clustered_ehnsw_engine_jobs;
	std::vector<job<disk_ehnsw_engine<float>, disk_ehnsw_engine_config>>
			disk_ehnsw_engine_jobs;
	std::vector<job<projection_engine<float, ehnsw_engine_2<float>>,
									projection_engine_config<ehnsw_engine_2_config>>>
			projection_ehnsw_engine_2_jobs;
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

	using namespace std::chrono_literals;

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
	if (false) {
		// for (size_t k : {28, 50}) {
		// for (size_t k : {55, 74, 80}) {
		// for (size_t k = 44; k <= 80; k += 12) {
		for (size_t k = 30; k <= 42; k += 6) {
			// for (size_t k = 70; k <= 95; k += 6) {
			// for (size_t num_for_1nn = 128; num_for_1nn <= 128; num_for_1nn *= 2) {
			// for (size_t num_for_1nn = 2; num_for_1nn <= 8; num_for_1nn *= 2) {
			// for (size_t num_for_1nn = 3; num_for_1nn <= 8; num_for_1nn += 1) {
			for (size_t num_for_1nn = 4; num_for_1nn <= 8; num_for_1nn *= 2) {
				// for (size_t num_for_1nn = 2; num_for_1nn <= 8; num_for_1nn *= 2) {
				//  for (size_t num_for_1nn = 2; num_for_1nn <= 128; num_for_1nn *= 2) {
				//   std::cerr << "About to start hnsw2(k=" << k << ",n4nn=" <<
				//   num_for_1nn
				//					<< ")" << std::endl;
				hnsw_engine_2_jobs.emplace_back(
						hnsw_engine_2_config(100, k, num_for_1nn, true));
				// projection_hnsw_engine_2_jobs.emplace_back(projection_engine_config(
				//		1, ds.dim, true, hnsw_engine_2_config(100, k, num_for_1nn,
				// true)));
				// projection_hnsw_engine_2_jobs.emplace_back(projection_engine_config(
				//		4, ds.dim, true, hnsw_engine_2_config(100, k, num_for_1nn,
				// true)));
				// projection_hnsw_engine_2_jobs.emplace_back(projection_engine_config(
				//		4, ds.dim / 4, true,
				//		hnsw_engine_2_config(100, k, num_for_1nn, true)));
				// projection_hnsw_engine_2_jobs.emplace_back(projection_engine_config(
				//		4, ds.dim / 6, true,
				//		hnsw_engine_2_config(100, k, num_for_1nn, true)));
				//    hnsw_engine_2<float> engine2(hnsw_engine_2_config(100, k,
				//    num_for_1nn));
				//    bdm.add(basic_benchmarker.get_benchmark_data(engine2));
				//     auto bd = basic_benchmarker.get_benchmark_data(engine2,
				//     default_timeout); std::string res_str =
				//     std::holds_alternative<bench_data>(bd) ?
				//     std::get<bench_data>(bd).to_string() 													:
				//     std::get<std::string>(bd);
				//     bdm.add(bd);
				//    std::cerr << "Completed hnsw2(k=" << k << ")" << std::endl;
				//     std::cerr << "Completed hnsw2(k=" << k << ") bd=" << res_str <<
				//     std::endl;
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
	if (false) {
		// for (size_t degree_cluster = 2; degree_cluster <= 64; degree_cluster *=
		// 4) {
		// for (size_t degree_cluster = 2; degree_cluster <= 2; degree_cluster *= 4)
		// {
		for (size_t degree_cluster = 8; degree_cluster <= 32; degree_cluster *= 4) {
			// for (size_t degree_cluster = 2; degree_cluster <= 2; degree_cluster *=
			// 4) { for (size_t degree_node = 2; degree_node <= 64; degree_node *= 4)
			// {
			for (size_t degree_node = 30; degree_node <= 42; degree_node += 6) {
				for (size_t num_for_1nn = 4; num_for_1nn <= 8; num_for_1nn *= 2) {
					hyper_hnsw_engine_jobs.emplace_back(hyper_hnsw_engine_config(
							100, degree_cluster, degree_node, num_for_1nn));
				}
			}
		}
	}

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
		// for (size_t k : {28, 50}) {
		// for (size_t k : {55, 74, 80}) {
		// for (size_t k : {85, 95}) {
		// for (size_t k = 44; k <= 80; k += 12) {
		// for (size_t k = 70; k <= 95; k += 6) {
		// for (size_t k = 40; k <= 40; k += 6) {
		// for (size_t k = 88; k <= 88; k += 6) {
		// for (size_t max_depth : {1, 2, 4, 20, 50, 100})
		// for (size_t max_depth : {1, 20, 50, 100})
		for (size_t max_depth : {1})
			// for (size_t k = 50; k <= 64; k += 12) {
			// for (size_t k = 50; k <= 86; k += 9) {
			// for (size_t k = 100; k <= 100; k += 20) {
			// for (size_t k = 60; k <= 200; k += 24) {
			// for (size_t k = 160; k <= 300; k += 12) {
			for (size_t k = 80; k <= 100; k += 5) {
				// for (size_t k = 100; k <= 100; k += 10) {
				// for (size_t num_for_1nn = 2; num_for_1nn <= 4; num_for_1nn *= 2) {
				// for (size_t num_for_1nn = 10; num_for_1nn <= 10; num_for_1nn *= 2) {
				for (size_t num_for_1nn = 1; num_for_1nn <= 32; num_for_1nn *= 2) {
					// for (size_t num_for_1nn = 20; num_for_1nn <= 20; num_for_1nn *= 2)
					// {
					if (false) {
						if (max_depth > 50) {
							hnsw_engine_2_jobs.emplace_back(
									hnsw_engine_2_config(max_depth, k, num_for_1nn, true));
						}
					}
					if (false) {
						hnsw_engine_2_jobs.emplace_back(
								hnsw_engine_2_config(100, k, num_for_1nn, true));
					}
					// for (size_t K = k - 40; K <= k; K += 12) {
					for (size_t K = k - 1; K <= k; K += 4) {
						for (size_t min_per_cut : {1}) {
							if (true) {
								// for (size_t elabel_prob_den = 2; elabel_prob_den < 80;
								//		 elabel_prob_den += 4)
								size_t elabel_prob_num = 1;
								// for (float elabel_prob_den = elabel_prob_num * 2;
								//		 elabel_prob_den < 240; elabel_prob_den += 3,
								//					 (elabel_prob_den > 40 && (elabel_prob_den += 5))) {
								for (float elabel_prob_den = elabel_prob_num * 2;
										 elabel_prob_den <= 2; elabel_prob_den += 3) {
									ehnsw_engine_2_jobs.emplace_back(ehnsw_engine_2_config(
											max_depth, k, num_for_1nn, K, min_per_cut, true, true,
											false, float(elabel_prob_num) / float(elabel_prob_den)));
								}
							}
							if (true) {
								size_t elabel_prob_num = 1;
								for (float elabel_prob_den = elabel_prob_num * 2;
										 elabel_prob_den <= 2; elabel_prob_den += 3) {
									/*
									for (size_t search_seed_width = 0; search_seed_width <= 4;
											 search_seed_width++) {
											 */
									for (size_t cut_run_length = 1; cut_run_length <= 1;
											 cut_run_length *= 2)
										ehnsw_engine_3_jobs.emplace_back(ehnsw_engine_3_config(
												max_depth, k, num_for_1nn, K, min_per_cut, true, true,
												false, float(elabel_prob_num) / float(elabel_prob_den),
												cut_run_length));
									// search_seed_width));
									//}
								}
							}
							if (false) {
								// for (float prune_coeff = 1.0f; prune_coeff < 200.0f;
								//		 prune_coeff += 10.0f) {
								for (float prune_coeff = 1.0f; prune_coeff < 18.0f;
										 prune_coeff += 0.4f) {
									jamana_ehnsw_engine_jobs.emplace_back(
											jamana_ehnsw_engine_config(max_depth, k, num_for_1nn, K,
																								 min_per_cut, true, true, false,
																								 0.5f, prune_coeff));
								}
							}
							if (false) {
								filter_ehnsw_engine_jobs.emplace_back(
										filter_ehnsw_engine_config(max_depth, k, num_for_1nn, K,
																							 min_per_cut, true, true, false));
							}
						}
					}
				}
			}
		for (size_t k = 38; k <= 60; k += 6) {
			// for (size_t k = 5; k <= 5; k += 6) {
			// for (size_t k : {28}) {
			// for (size_t num_for_1nn = 2; num_for_1nn <= 8; num_for_1nn *= 2) {
			// for (size_t num_for_1nn = 4; num_for_1nn <= 8; num_for_1nn *= 2) {
			// for (size_t num_for_1nn = 2; num_for_1nn <= 8; num_for_1nn *= 2) {
			for (size_t num_for_1nn = 2; num_for_1nn <= 2; num_for_1nn *= 2) {
				// for (size_t num_for_1nn = 3; num_for_1nn <= 8; num_for_1nn += 1) {
				// for (size_t num_for_1nn = 4; num_for_1nn <= 4; num_for_1nn *= 2) {
				// for (size_t num_for_1nn = 64; num_for_1nn <= 128; num_for_1nn *= 2) {
				// for (size_t num_for_1nn = 2; num_for_1nn <= 2; num_for_1nn *= 2) {
				//  for (size_t K : {2, 4}) {
				// for (size_t K : {2, 4, 8}) {
				// for (size_t K : {1, 3}) {
				// for (size_t K : {1}) {
				for (size_t K : {3}) {
					// for (size_t min_per_cut : {1, 2}) {
					for (size_t min_per_cut : {1}) {
						// std::cerr << "About to start ehnsw2(k=" << k << ",K=" << K
						//					<< ",n4nn=" << num_for_1nn
						//					<< ",min_per_cut=" << min_per_cut << ")" << std::endl;
						// ehnsw_engine_2_jobs.emplace_back(ehnsw_engine_2_config(
						//		100, k, num_for_1nn, K, min_per_cut, true, true, true));
						// ehnsw_engine_2_jobs.emplace_back(ehnsw_engine_2_config(
						//		100, k, num_for_1nn, K, min_per_cut, true, true));
						if (false) {
							ehnsw_engine_2_jobs.emplace_back(ehnsw_engine_2_config(
									100, k, num_for_1nn, K, min_per_cut, true, true, false));
						}
						if (false) {
							// size_t cluster_size = 1;
							// size_t min_cluster_membership = 1;
							for (size_t cluster_size = 1; cluster_size <= 16;
									 cluster_size *= 2) {
								for (size_t min_cluster_membership = cluster_size;
										 min_cluster_membership <= 16;
										 min_cluster_membership *= 2) {
									clustered_ehnsw_engine_jobs.emplace_back(
											clustered_ehnsw_engine_config(
													100, k, num_for_1nn, K, min_per_cut, cluster_size,
													min_cluster_membership, true, true, false));
								}
							}
						}
						if (false) {
							bool create_index = true;
							disk_ehnsw_engine_jobs.emplace_back(disk_ehnsw_engine_config(
									ehnsw_engine_2_config(100, k, num_for_1nn, K, min_per_cut,
																				true, true, false),
									"tempindexfile.expannind", create_index));
						}
						//  ehnsw_engine_2_jobs.emplace_back(ehnsw_engine_2_config(
						//		100, k, num_for_1nn, K, min_per_cut, true, false));
						//   ehnsw_engine_2<float> engine(
						//		ehnsw_engine_2_config(100, k, num_for_1nn, K, min_per_cut));
						//   bdm.add(basic_benchmarker.get_benchmark_data(engine));
						//   std::cerr << "Completed ehnsw2(k=" << k << ",K=" << K
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
		// for (size_t tc = 32; tc <= 48; tc += 8) {
		for (size_t tc = 8; tc <= 32; tc *= 2) {
			// for (size_t max_leaf_size = 64; max_leaf_size <= 1024 * 4;
			// max_leaf_size *= 4) {
			for (size_t max_leaf_size = 1024; max_leaf_size <= 1024 * 4 * 4;
					 max_leaf_size *= 16) {
				for (size_t sc = max_leaf_size / tc; sc * tc <= 8192 * 1; sc *= 16) {
					for (size_t num_probes = 0; num_probes <= 20; num_probes += 4) {
						// for (size_t sc = max_leaf_size; sc * tc <= 8192 * 8; sc *= 16 *
						// 2)
						// {
						//  std::cerr << "Starting tree arrangement(tc=" << tc
						//					<< ",max_leaf_size=" << max_leaf_size << ",sc=" << sc <<
						//")"
						//					<< std::endl;
						//  std::cerr << "Expected time proportional to: " << sc * tc
						//					<< std::endl;
						//  auto begin = std::chrono::high_resolution_clock::now();
						tree_arrangement_engine_jobs.emplace_back(
								tree_arrangement_engine_config(tc, max_leaf_size, sc, 3, 8, 40,
																							 num_probes));
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
	return perform_and_store_benchmark_results(
			ds.name, num_threads, basic_benchmarker, hnsw_engine_jobs,
			hnsw_engine_2_jobs, arrangement_engine_jobs, ehnsw_engine_jobs,
			ehnsw_engine_2_jobs, ehnsw_engine_3_jobs, jamana_ehnsw_engine_jobs,
			filter_ehnsw_engine_jobs, clustered_ehnsw_engine_jobs,
			hier_arrangement_engine_jobs, hnsw_engine_hybrid_jobs,
			tree_arrangement_engine_jobs, tree_arrangement_engine_if_jobs,
			projection_hnsw_engine_2_jobs, projection_ehnsw_engine_2_jobs,
			disk_ehnsw_engine_jobs, hyper_hnsw_engine_jobs);
	/*
	perform_benchmarks_with_threads(
			basic_benchmarker, num_threads, hnsw_engine_jobs, hnsw_engine_2_jobs,
			arrangement_engine_jobs, ehnsw_engine_jobs, ehnsw_engine_2_jobs,
			ehnsw_engine_3_jobs, jamana_ehnsw_engine_jobs, filter_ehnsw_engine_jobs,
			clustered_ehnsw_engine_jobs, hier_arrangement_engine_jobs,
			hnsw_engine_hybrid_jobs, tree_arrangement_engine_jobs,
			tree_arrangement_engine_if_jobs, projection_hnsw_engine_2_jobs,
			projection_ehnsw_engine_2_jobs, disk_ehnsw_engine_jobs,
			hyper_hnsw_engine_jobs);

	bench_data_manager bdm(ds.name);
	store_benchmark_results(
			bdm, hnsw_engine_jobs, hnsw_engine_2_jobs, arrangement_engine_jobs,
			ehnsw_engine_jobs, ehnsw_engine_2_jobs, ehnsw_engine_3_jobs,
			jamana_ehnsw_engine_jobs, filter_ehnsw_engine_jobs,
			clustered_ehnsw_engine_jobs, hier_arrangement_engine_jobs,
			hnsw_engine_hybrid_jobs, tree_arrangement_engine_jobs,
			tree_arrangement_engine_if_jobs, projection_hnsw_engine_2_jobs,
			projection_ehnsw_engine_2_jobs, disk_ehnsw_engine_jobs,
			hyper_hnsw_engine_jobs);

	return bdm;
	*/
}
