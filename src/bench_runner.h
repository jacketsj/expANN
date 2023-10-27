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
#include "ehnsw_engine_4.h"
#include "ehnsw_engine_5.h"
#include "ehnsw_engine_6.h"
#include "ehnsw_engine_7.h"
#include "ehnsw_engine_8.h"
#include "ehnsw_engine_basic.h"
#include "ehnsw_engine_basic_pqn.h"
#include "ensg_engine.h"
#include "filter_ehnsw_engine.h"
#include "hier_arrangement_engine.h"
#include "hnsw_engine.h"
#include "hnsw_engine_2.h"
#include "hnsw_engine_basic_2.h"
#include "hnsw_engine_basic_3.h"
#include "hnsw_engine_basic_4.h"
#include "hnsw_engine_basic_clustered.h"
#include "hnsw_engine_hybrid.h"
#include "hnsw_engine_reference.h"
#include "hyper_hnsw_engine.h"
#include "isect_clustering_engine.h"
#include "jamana_ehnsw_engine.h"
#include "projection_engine.h"
#include "static_rcg_engine.h"
#include "static_rcg_engine_simple.h"
#include "tree_arrangement_engine.h"
#include "tree_arrangement_engine_if.h"
#include "zehnsw_engine.h"

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

	std::vector<job<hnsw_engine<float, false>, hnsw_engine_config>>
			hnsw_engine_jobs;
	std::vector<job<hnsw_engine_2<float>, hnsw_engine_2_config>>
			hnsw_engine_2_jobs;
	std::vector<job<hnsw_engine_basic_2<float>, hnsw_engine_basic_2_config>>
			hnsw_engine_basic_2_jobs;
	std::vector<job<hnsw_engine_basic_3<float>, hnsw_engine_basic_3_config>>
			hnsw_engine_basic_3_jobs;
	std::vector<job<hnsw_engine_basic_4<float>, hnsw_engine_basic_4_config>>
			hnsw_engine_basic_4_jobs;
	std::vector<job<ehnsw_engine_basic<float>, ehnsw_engine_basic_config>>
			ehnsw_engine_basic_jobs;
	std::vector<job<ehnsw_engine_basic_pqn<float>, ehnsw_engine_basic_pqn_config>>
			ehnsw_engine_basic_pqn_jobs;
	std::vector<job<hnsw_engine_basic_clustered<float>,
									hnsw_engine_basic_clustered_config>>
			hnsw_engine_basic_clustered_jobs;
	std::vector<job<static_rcg_engine<float>, static_rcg_engine_config>>
			static_rcg_engine_jobs;
	std::vector<
			job<static_rcg_engine_simple<float>, static_rcg_engine_simple_config>>
			static_rcg_engine_simple_jobs;
	std::vector<job<hnsw_engine_reference<float>, hnsw_engine_reference_config>>
			hnsw_engine_reference_jobs;
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
	std::vector<job<ehnsw_engine_4<float>, ehnsw_engine_4_config>>
			ehnsw_engine_4_jobs;
	std::vector<job<ehnsw_engine_5<float>, ehnsw_engine_5_config>>
			ehnsw_engine_5_jobs;
	std::vector<job<ehnsw_engine_6<float>, ehnsw_engine_6_config>>
			ehnsw_engine_6_jobs;
	std::vector<job<ehnsw_engine_7<float>, ehnsw_engine_7_config>>
			ehnsw_engine_7_jobs;
	std::vector<job<ehnsw_engine_8<float>, ehnsw_engine_8_config>>
			ehnsw_engine_8_jobs;
	std::vector<job<zehnsw_engine<float>, zehnsw_engine_config>>
			zehnsw_engine_jobs;
	std::vector<job<ensg_engine<float>, ensg_engine_config>> ensg_engine_jobs;
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
	std::vector<
			job<isect_clustering_engine<float>, isect_clustering_engine_config>>
			isect_clustering_engine_jobs;

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
	if (true) {
		// for (size_t k = 80; k <= 120; k += 5) {
		// for (size_t k = 80; k <= 200; k += 40) {
		// for (size_t k = 100; k <= 100; k += 40) {
		// for (size_t k = 40; k <= 40; k += 40) {
		// for (size_t k = 120; k <= 120; k += 20) {
		// for (size_t k = 200; k <= 200; k += 20) {
		// for (size_t k = 20; k <= 20; k += 20) { // used for small opt stuff, want
		//																				// this to be very good on its own
		// for (size_t k = 120; k <= 120; k += 20) { // gets decent recall already
		// for (size_t k = 100; k <= 140; k += 20) { // gets very good recall and
		// range
		// for (size_t k = 50; k <= 70; k += 10) { // will hopfully get good recall?
		// for (size_t k = 60; k <= 60; k += 10) { // will hopfully get good
		// for (size_t k = 16; k <= 16; k += 10) { // will hopfully get good
		// for (size_t k = 32; k <= 32; k += 10) { // will hopfully get good
		// for (size_t k = 16; k <= 16; k += 10) { // will hopfully get good
		// for (size_t k = 20; k <= 20; k += 20) { // will hopfully get good
		// for (size_t k = 70; k <= 70; k += 20) { // will hopfully get good
		// for (size_t k = 50; k <= 70; k += 20) { // will hopfully get good
		// for (size_t k = 30; k <= 30; k += 20) { // will hopfully get good
		// for (size_t k = 30; k <= 50; k += 20) { // will hopfully get good
		// for (size_t k = 55; k <= 55; k += 20) { // will hopfully get good
		// for (size_t k = 30; k <= 60; k += 10) { // will hopfully get good
		// for (size_t k = 40; k <= 40; k += 10) { // will hopfully get good
		for (size_t k = 40; k <= 40; k += 10) { // will hopfully get good
			// for (size_t k = 90; k <= 90; k += 7) { // will hopfully get good
			// for (size_t k = 30; k <= 30; k += 7) { // will hopfully get good
			// for (size_t k = 40; k <= 40; k += 7) { // will hopfully get good
			// for (size_t k = 10; k <= 10; k += 7) { // will hopfully get good
			//  recall?
			//   for (size_t k = 20; k <= 50; k += 10) {
			//   for (size_t k = 40; k <= 50; k += 10) {
			//   for (size_t k = 100; k <= 100; k += 20) {
			//    for (size_t k = 80; k <= 100; k += 20) {
			//    for (size_t k = 80; k <= 80; k += 20) {
			//    for (size_t k = 100; k <= 100; k += 20) {
			//    for (size_t k = 50; k <= 50; k += 20) {
			//    for (size_t num_for_1nn = 2; num_for_1nn <= 32; num_for_1nn *= 4) {
			//    for (size_t num_for_1nn = 2; num_for_1nn <= 16; num_for_1nn *= 2) {
			//    for (size_t num_for_1nn = 1; num_for_1nn <= 6; num_for_1nn += 1) {
			//    for (size_t num_for_1nn = 3; num_for_1nn <= 4; num_for_1nn += 1) {
			//   for (size_t num_for_1nn = 4; num_for_1nn <= 4; num_for_1nn += 1) {
			//   for (size_t num_for_1nn = 16; num_for_1nn <= 64; num_for_1nn *= 2) {
			//   for (size_t num_for_1nn = 4; num_for_1nn <= 16; num_for_1nn *= 2) {
			//   for (size_t num_for_1nn = 4; num_for_1nn <= 4; num_for_1nn *= 2) {
			//  for (size_t num_for_1nn = 4; num_for_1nn <= 4; num_for_1nn *= 2) {
			//  for (size_t num_for_1nn = 4; num_for_1nn <= 4; num_for_1nn *= 2) {
			//  for (size_t num_for_1nn = 4; num_for_1nn <= 16; num_for_1nn *= 2) {
			//  for (size_t num_for_1nn = 1; num_for_1nn <= 4; num_for_1nn *= 2) {
			//  for (size_t num_for_1nn = 1; num_for_1nn <= 16; num_for_1nn *= 4) {
			//  for (size_t num_for_1nn = 2; num_for_1nn <= 2; num_for_1nn *= 4) {
			//  for (size_t num_for_1nn = 16; num_for_1nn <= 16; num_for_1nn *= 4) {
			// for (size_t num_for_1nn = 1; num_for_1nn <= 1; num_for_1nn *= 4) {
			// for (size_t num_for_1nn = 10; num_for_1nn <= 10; num_for_1nn *= 2) {
			// for (size_t num_for_1nn = 1; num_for_1nn <= 4; num_for_1nn *= 2) {
			for (size_t num_for_1nn = 1; num_for_1nn <= 1; num_for_1nn *= 2) {
				if (false) {
					ensg_engine_jobs.emplace_back(
							ensg_engine_config(k, num_for_1nn, 1.0f));
				}
				if (false) {
					ehnsw_engine_2_jobs.emplace_back(ehnsw_engine_2_config(
							100, k, num_for_1nn, k - 1, 1, true, true, false, 0.5f));
				}
				if (false) {
					for (size_t edge_count_search_factor : {4}) {
						ehnsw_engine_7_jobs.emplace_back(ehnsw_engine_7_config(
								100, k, k * edge_count_search_factor, num_for_1nn, k - 1, 1,
								true, true, false, 0.5f));
					}
				}
				if (false) {
					for (size_t edge_count_search_factor : {1}) {
						hnsw_engine_basic_2_jobs.emplace_back(hnsw_engine_basic_2_config(
								k, 2 * k, num_for_1nn, k * edge_count_search_factor));
					}
				}
				if (false) {
					for (size_t edge_count_search_factor : {1}) {
						hnsw_engine_basic_3_jobs.emplace_back(hnsw_engine_basic_3_config(
								k, 2 * k, num_for_1nn, k * edge_count_search_factor));
					}
				}
				if (false) {
					for (size_t edge_count_search_factor : {2}) {
						hnsw_engine_basic_4_jobs.emplace_back(hnsw_engine_basic_4_config(
								k, 2 * k, num_for_1nn, k * edge_count_search_factor));
					}
				}
				if (false) {
					for (size_t edge_count_search_factor : {2}) {
						ehnsw_engine_basic_jobs.emplace_back(ehnsw_engine_basic_config(
								k, 2 * k, num_for_1nn, k * edge_count_search_factor));
					}
				}
				if (true) {
					for (size_t edge_count_search_factor : {2}) {
						ehnsw_engine_basic_pqn_jobs.emplace_back(
								ehnsw_engine_basic_pqn_config(k, 2 * k, num_for_1nn,
																							k * edge_count_search_factor, 14,
																							16, 8));
					}
				}
				if (false) {
					for (size_t edge_count_search_factor : {1}) {
						zehnsw_engine_jobs.emplace_back(
								zehnsw_engine_config(k, num_for_1nn, edge_count_search_factor));
					}
				}
				if (false) {
					for (size_t edge_count_search_factor : {1}) {
						for (size_t cluster_size : {4, 16, 32}) {
							for (size_t cluster_neighbours : {0, 4, 8, 32}) {
								hnsw_engine_basic_clustered_jobs.emplace_back(
										hnsw_engine_basic_clustered_config(
												k, 2 * k, num_for_1nn, k * edge_count_search_factor,
												cluster_size, cluster_neighbours));
							}
						}
					}
				}
				if (false) {
					for (size_t edge_count_search_factor : {1}) {
						for (size_t MM : {3})
							static_rcg_engine_simple_jobs.emplace_back(
									static_rcg_engine_simple_config(
											k, 2 * k, MM, num_for_1nn, k * edge_count_search_factor));
					}
				}
				if (false) {
					for (size_t edge_count_search_factor : {1}) {
						for (size_t cluster_overlap : {1}) {
							// for (size_t C : {8 * 8 * 8 * 2}) { // 4 * k * cluster_overlap
							for (size_t C :
									 {8}) { //, 8 * 8 * 2}) { // 4 * k * cluster_overlap
								for (size_t rC : {k}) {
									for (size_t brute_force_size :
											 {2 * k * edge_count_search_factor + 2}) {
										//{k * C * edge_count_search_factor + 1}) {
										static_rcg_engine_jobs.emplace_back(
												static_rcg_engine_config(k, cluster_overlap, C, rC,
																								 brute_force_size, num_for_1nn,
																								 k * edge_count_search_factor));
									}
								}
							}
						}
					}
				}
				if (false) {
					// for (size_t edge_count_search_factor : {4, 8}) {
					for (size_t edge_count_search_factor : {1}) {
						// for (bool use_ecuts : {false, true}) {
						for (bool use_ecuts : {false}) {
							hnsw_engine_reference_jobs.emplace_back(
									hnsw_engine_reference_config(k, edge_count_search_factor * k,
																							 num_for_1nn, use_ecuts));
						}
					}
				}
				if (false) {
					for (size_t edge_count_search_factor : {1}) {
						ehnsw_engine_8_jobs.emplace_back(ehnsw_engine_8_config(
								k, num_for_1nn, edge_count_search_factor));
					}
				}
				if (false) {
					for (size_t edge_count_search_factor : {1}) {
						for (bool extend_to_neighbours : {false}) {
							ehnsw_engine_6_jobs.emplace_back(ehnsw_engine_6_config(
									k, num_for_1nn, edge_count_search_factor * k,
									extend_to_neighbours));
						}
					}
				}
				if (false) {
					// for (size_t edge_count_search_factor : {2, 4}) {
					// for (size_t edge_count_search_factor : {4}) {
					// for (size_t edge_count_search_factor : {4}) {
					for (size_t edge_count_search_factor : {1}) {
						// for (double layer_multiplier :
						//		 {1 / log(k), 2 / log(k), 4 / log(k), 0.3})
						//	for (double layer_multiplier = 0.2; layer_multiplier <= 2.0f;
						//			 layer_multiplier += 0.2f)
						ehnsw_engine_5_jobs.emplace_back(ehnsw_engine_5_config(
								k, num_for_1nn, edge_count_search_factor));
						//, layer_multiplier));
					}
				}
				if (false) {
					// for (size_t edge_count_search_factor = 1;
					//		 edge_count_search_factor <= 4; ++edge_count_search_factor)
					// for (size_t edge_count_search_factor : {1, 4, 8})
					for (size_t edge_count_search_factor : {1})
						ehnsw_engine_4_jobs.emplace_back(ehnsw_engine_4_config(
								k, num_for_1nn, 100, 1.0f, false, false, false, 1, false, false,
								1.0f, edge_count_search_factor));
					// ehnsw_engine_4_jobs.emplace_back(ehnsw_engine_4_config(
					//		k, num_for_1nn, 100, 1.0f, true, true, false, 4, true));
				}
				if (false) {
					for (bool use_pruning : {true}) {
						for (float pruning_factor : {1.0f, 1.1f, 1.2f, 1.8f}) {
							ehnsw_engine_4_jobs.emplace_back(ehnsw_engine_4_config(
									k, num_for_1nn, 100, 1.0f, true, true, false, 4, true,
									use_pruning, pruning_factor));
						}
					}
				}
				if (false) {
					// for (bool include_visited_during_build : {false}) {
					for (bool include_visited_during_build : {false, true}) {
						// for (bool run_improves : {false}) {
						for (bool run_improves : {false, true}) {
							size_t cut_off_visited_if_long_ratio = 4;
							if (include_visited_during_build) {
								for (bool cut_off_visited_if_long : {false, true}) {
									for (bool include_visited_only_higher : {false, true}) {
										ehnsw_engine_4_jobs.emplace_back(ehnsw_engine_4_config(
												k, num_for_1nn, 100, 1.0f, include_visited_during_build,
												run_improves, cut_off_visited_if_long,
												cut_off_visited_if_long_ratio,
												include_visited_only_higher));
									}
								}
							} else {
								ehnsw_engine_4_jobs.emplace_back(ehnsw_engine_4_config(
										k, num_for_1nn, 100, 1.0f, include_visited_during_build,
										run_improves, false, cut_off_visited_if_long_ratio));
							}
						}
					}
				}
				if (false) {
					ehnsw_engine_2_jobs.emplace_back(ehnsw_engine_2_config(
							100, k, num_for_1nn, 3, 1, true, true, false, 0.5f));
					ehnsw_engine_2_jobs.emplace_back(ehnsw_engine_2_config(
							1, k, num_for_1nn, k - 1, 1, true, true, false, 0.5f));
				}
				if (false) {
					bool bumping = true;
					for (size_t layers : {1, 100})
						for (float initial_random_branch_prob = 0.4f;
								 initial_random_branch_prob > 0.2f;
								 // initial_random_branch_prob > 0.0001f;
								 initial_random_branch_prob /= 2.0f) {
							for (float random_branch_decay = initial_random_branch_prob;
									 random_branch_decay >= 0.000001f;
									 random_branch_decay /= 16.0f) {
								ehnsw_engine_3_jobs.emplace_back(ehnsw_engine_3_config(
										layers, k, num_for_1nn, k - 1, 1, true, bumping, false,
										0.5f, 1, initial_random_branch_prob, random_branch_decay));
							}
						}
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
							if (false) {
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
							if (false) {
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
						tree_arrangement_engine_jobs.emplace_back(
								tree_arrangement_engine_config(tc, max_leaf_size, sc, 3, 8, 40,
																							 num_probes));
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
					tree_arrangement_engine_if_jobs.emplace_back(
							tree_arrangement_engine_if_config(tc, max_leaf_size, sc));
				}
			}
		}
	}
	if (false) {
		// for (size_t tc = 2; tc <= 64; tc *= 2) {
		for (size_t tc = 64; tc <= 64; tc *= 2) {
			// for (size_t tc = 4; tc <= 4; tc *= 2) {
			for (size_t max_leaf_size = 64 * 2; max_leaf_size <= 1024 * 4;
					 max_leaf_size *= 4) {
				// for (size_t sc = max_leaf_size; sc * tc <= 8192 * 8; sc *= 16 * 2) {
				for (size_t sc = max_leaf_size; sc <= 64 * 64 * 8 * 16; sc *= 16) {
					// for (size_t num_isect = 1; num_isect <= 8; ++num_isect) {
					for (size_t num_isect = 8; num_isect <= 32; num_isect += 8) {
						// for (size_t num_isect = 4; num_isect <= 8; num_isect += 1) {
						for (size_t cluster_overlap = 1; cluster_overlap <= 1;
								 cluster_overlap *= 2) {
							isect_clustering_engine_jobs.emplace_back(
									isect_clustering_engine_config(tc, max_leaf_size, sc, 8,
																								 num_isect, 40,
																								 cluster_overlap));
						}
					}
				}
			}
		}
	}
	return perform_and_store_benchmark_results(
			ds.name, num_threads, basic_benchmarker, hnsw_engine_jobs,
			hnsw_engine_2_jobs, hnsw_engine_basic_2_jobs, hnsw_engine_basic_3_jobs,
			hnsw_engine_basic_4_jobs, ehnsw_engine_basic_jobs,
			ehnsw_engine_basic_pqn_jobs, hnsw_engine_basic_clustered_jobs,
			static_rcg_engine_jobs, static_rcg_engine_simple_jobs,
			hnsw_engine_reference_jobs, arrangement_engine_jobs, ehnsw_engine_jobs,
			ehnsw_engine_2_jobs, ehnsw_engine_3_jobs, ehnsw_engine_4_jobs,
			ehnsw_engine_5_jobs, ehnsw_engine_6_jobs, ehnsw_engine_7_jobs,
			ehnsw_engine_8_jobs, zehnsw_engine_jobs, ensg_engine_jobs,
			jamana_ehnsw_engine_jobs, filter_ehnsw_engine_jobs,
			clustered_ehnsw_engine_jobs, hier_arrangement_engine_jobs,
			hnsw_engine_hybrid_jobs, tree_arrangement_engine_jobs,
			tree_arrangement_engine_if_jobs, isect_clustering_engine_jobs,
			projection_hnsw_engine_2_jobs, projection_ehnsw_engine_2_jobs,
			disk_ehnsw_engine_jobs, hyper_hnsw_engine_jobs);
}
