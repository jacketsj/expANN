#pragma once

#include <string>
#include <vector>

#include "basic_bench.h"
#include "bench_data_manager.h"

//#include "arrangement_engine.h"
#include "brute_force_engine.h"
#include "dataset.h"
#include "dataset_loader.h"
//#include "ehnsw_engine.h"
#include "ehnsw_engine_2.h"
//#include "hier_arrangement_engine.h"
//#include "hnsw_engine.h"
#include "hnsw_engine_2.h"
#include "hnsw_engine_3.h"
//#include "hnsw_engine_hybrid.h"
#include "tree_arrangement_engine.h"
#include "tree_arrangement_engine_if.h"

template <typename test_dataset_t>
bench_data_manager perform_benchmarks(test_dataset_t ds) {
	basic_bench<float, test_dataset_t> basic_benchmarker(ds);
	bench_data_manager bdm(ds.name);

	// TODO implement a way to disable the timeout in get_benchmark_data (e.g. if
	// it's 0, or at least a way to make it super long)
	std::cerr << "Running benchmarks with a 6000s timeout" << std::endl;

	using namespace std::chrono_literals;
	auto default_timeout = 6000s;

	brute_force_engine<float> engine_bf;
	bdm.add(basic_benchmarker.get_benchmark_data(engine_bf, default_timeout));

	// if (false) {
	//	for (size_t k = 100; k <= 140; k += 40) {
	//		for (int p2 = 15; p2 < 18; ++p2) {
	//			if (p2 > 4)
	//				p2 += 2;
	//			std::cerr << "About to start hnsw(k=" << k << ",p2=" << p2 << ")"
	//								<< std::endl;
	//			hnsw_engine<float, false> engine(50, k, 0.5 * p2);
	//			bdm.add(basic_benchmarker.get_benchmark_data(engine,
	// default_timeout)); 			std::cerr << "Completed hnsw(k=" << k << ",p2=" <<
	// p2
	// <<
	//")"
	//								<< std::endl;
	//		}
	//	}
	//	// for (size_t k = 2; k <= 128; k += 12) {
	//	// 	for (size_t num_for_1nn = 32; num_for_1nn <= 64 * 4 * 2; num_for_1nn
	//*=
	//	// 4)
	//	// {
	// }
	if (true) {
		for (size_t k = 38; k <= 50; k += 10) {
			for (size_t num_for_1nn = 1; num_for_1nn <= 27; num_for_1nn *= 3) {
				std::cerr << "About to start hnsw2(k=" << k << ",n4nn=" << num_for_1nn
									<< ")" << std::endl;
				hnsw_engine_2<float> engine2(100, k, num_for_1nn);
				hnsw_engine_3<float> engine3(100, k, num_for_1nn);
				bdm.add(basic_benchmarker.get_benchmark_data(engine2, default_timeout));
				bdm.add(basic_benchmarker.get_benchmark_data(engine3, default_timeout));
				std::cerr << "Completed hnsw2(k=" << k << ")" << std::endl;
			}
		}
	}
	struct ehnsw2_run {
		size_t edge_count_mult;
		size_t max_depth;
		size_t min_per_cut;
		size_t num_cuts;
		size_t num_for_1nn;
		ehnsw2_run(size_t _edge_count_mult, size_t _max_depth, size_t _min_per_cut,
							 size_t _num_cuts, size_t _num_for_1nn)
				: edge_count_mult(_edge_count_mult), max_depth(_max_depth),
					min_per_cut(_min_per_cut), num_cuts(_num_cuts),
					num_for_1nn(_num_for_1nn) {}
	};
	std::vector<ehnsw2_run> to_run;
	//	for (size_t ecm = 10; ecm <= 40; ecm *= 2)
	//		for (size_t mpc = 1; mpc <= 8; mpc *= 2)
	//			for (size_t nc = 1; nc <= 8; nc *= 2)
	//				for (size_t n4nn = 1; n4nn <= 16; n4nn *= 4)
	// for (size_t ecm = 2; ecm <= 10; ecm += 1)
	for (size_t ecm = 10; ecm <= 10; ecm += 1)
		for (size_t mpc = 1; mpc <= 4; mpc *= 2)
			for (size_t nc = 1; nc * mpc < ecm; nc *= 2)
				for (size_t n4nn = 16; n4nn <= 16; n4nn *= 4) {
					to_run.emplace_back(ecm, 100, mpc, nc, n4nn);
				}
	// to_run.emplace_back(56, 100, 1, 16, 32);
	// to_run.emplace_back(56, 100, 4, 8, 8);
	// to_run.emplace_back(47, 100, 1, 16, 64);
	// to_run.emplace_back(46, 100, 1, 4, 128);
	std::cerr << "Running " << to_run.size() << " jobs in total." << std::endl;
	std::vector<std::variant<bench_data, std::string>> to_run_bd(to_run.size());
	{
		std::vector<std::jthread> to_run_jobs;
		size_t num_threads = 6;
		for (size_t t_index = 0; t_index < num_threads; ++t_index) {
			to_run_jobs.emplace_back([&to_run_bd, &to_run, t_index,
																&basic_benchmarker, default_timeout,
																num_threads]() {
				for (size_t r_index = t_index; r_index < to_run.size();
						 r_index += num_threads) {
					// for (auto& r : to_run) {
					size_t k = to_run[r_index].edge_count_mult;
					size_t K = to_run[r_index].num_cuts;
					size_t md = to_run[r_index].max_depth;
					size_t mpc = to_run[r_index].min_per_cut;
					size_t n4nn = to_run[r_index].num_for_1nn;
					std::cerr << "About to start ehnsw2(k=" << k << ",K=" << K
										<< ",n4nn=" << n4nn << ",min_per_cut=" << mpc << ")"
										<< std::endl;
					ehnsw_engine_2<float> engine(md, k, n4nn, K, mpc);
					// bdm.add(basic_benchmarker.get_benchmark_data(engine,
					// default_timeout));
					to_run_bd[r_index] =
							basic_benchmarker.get_benchmark_data(engine, default_timeout);
					std::cerr << "Completed ehnsw2(k=" << k << ",K=" << K
										<< ",n4nn=" << n4nn << ",min_per_cut=" << mpc << ")"
										<< std::endl;
				}
			});
		}
	}
	for (size_t r_index = 0; r_index < to_run.size(); ++r_index)
		bdm.add(to_run_bd[r_index]);
	// for (size_t K = 4; K <= 64; K += 4) {
	//	for (size_t k = 4; k * K <= 128 * 8; k += 4) {
	//		for (size_t num_for_1nn = 32; num_for_1nn <= 64 * 4 * 2;
	// for (size_t K = 4; K <= 256; K *= 2) {
	//	for (size_t k = 16; k <= 128 * 2; k *= 4) {
	//		for (size_t num_for_1nn = 32 * 4; num_for_1nn <= 64 * 2;
	if (false) {
		for (size_t K = 2; K <= 32; K *= 2) {
			for (size_t k = 11; k < 64; k += 9) {
				for (size_t num_for_1nn = 4; num_for_1nn <= 64; num_for_1nn *= 2) {
					for (size_t min_per_cut = 1;
							 min_per_cut * K <= k && min_per_cut <= 16; min_per_cut *= 2) {
						std::cerr << "About to start ehnsw2(k=" << k << ",K=" << K
											<< ",n4nn=" << num_for_1nn
											<< ",min_per_cut=" << min_per_cut << ")" << std::endl;
						ehnsw_engine_2<float> engine(100, k, num_for_1nn, K, min_per_cut);
						bdm.add(
								basic_benchmarker.get_benchmark_data(engine, default_timeout));
						std::cerr << "Completed ehnsw2(k=" << k << ",K=" << K
											<< ",n4nn=" << num_for_1nn
											<< ",min_per_cut=" << min_per_cut << ")" << std::endl;
					}
				}
			}
		}
	}
	// if (false) {
	//	for (size_t K = 4; K <= 32; K += 8) {
	//		for (size_t k = 2; k * K <= 64; k += 12) {
	//			for (size_t num_for_1nn = 1; num_for_1nn <= 40; num_for_1nn *= 4) {
	//				std::cerr << "About to start ehnsw(k=" << k << ",K=" << K
	//									<< ",n4nn=" << num_for_1nn << ")" << std::endl;
	//				ehnsw_engine<float> engine(100, k, K, num_for_1nn);
	//				bdm.add(
	//						basic_benchmarker.get_benchmark_data(engine, default_timeout));
	//				std::cerr << "Completed ehnsw(k=" << k << ",K=" << K
	//									<< ",n4nn=" << num_for_1nn << ")" << std::endl;
	//			}
	//		}
	//	}
	// }
	// if (false) {
	//	for (size_t k = 2; k <= 60; k += 12) {
	//		for (size_t num_for_1nn = 1; num_for_1nn <= 40; num_for_1nn *= 4) {
	//			std::cerr << "About to start hnsw_hybrid(k=" << k
	//								<< ",n4nn=" << num_for_1nn << ")" << std::endl;
	//			hnsw_engine_hybrid<float> engine(100, k, num_for_1nn);
	//			bdm.add(basic_benchmarker.get_benchmark_data(engine,
	// default_timeout)); 			std::cerr << "Completed hnsw_hybrid(k=" << k <<
	// ",n4nn="
	//<< num_for_1nn
	//								<< ")" << std::endl;
	//		}
	//	}

	//	if (false) {
	//		for (size_t k = 1; k < 4; ++k) {
	//			for (size_t n = 4; n <= 256; n *= 2) {
	//				for (size_t m = 1; m <= 64; m *= 4) {
	//					std::cerr << "Starting arrangement(k=" << k << ",n=" << n
	//										<< ",m=" << m << ")" << std::endl;
	//					arrangement_engine<float> engine(k, n, m);
	//					bdm.add(
	//							basic_benchmarker.get_benchmark_data(engine,
	// default_timeout)); 					std::cerr << "Completed arrangement(k=" << k
	// <<
	// ",n=" <<
	// n
	//										<< ",m=" << m << ")" << std::endl;
	//				}
	//			}

	//			for (size_t na = 1; na <= 16; na *= 4) {
	//				for (size_t levels = 1; levels * na <= 32; levels *= 2) {
	//					for (size_t sc = 32; sc <= 8192; sc *= 4) {
	//						std::cerr << "Starting hier arrangement(na=" << na
	//											<< ",levels=" << levels << ",sc=" << sc << ")"
	//											<< std::endl;
	//						hier_arrangement_engine<float> engine(na, levels, sc);
	//						bdm.add(basic_benchmarker.get_benchmark_data(engine,
	//																												 default_timeout));
	//						std::cerr << "Completed hier arrangement(na=" << na
	//											<< ",levels=" << levels << ",sc=" << sc << ")"
	//											<< std::endl;
	//					}
	//				}
	//			}
	//		}
	//	}
	//}
	if (true) {
		for (size_t tc = 2; tc <= 64; tc *= 2) {
			for (size_t max_leaf_size = 64; max_leaf_size <= 1024 * 4;
					 max_leaf_size *= 4) {
				for (size_t sc = max_leaf_size; sc * tc <= 8192 * 8; sc *= 16 * 2) {
					std::cerr << "Starting tree arrangement(tc=" << tc
										<< ",max_leaf_size=" << max_leaf_size << ",sc=" << sc << ")"
										<< std::endl;
					std::cerr << "Expected time proportional to: " << sc * tc
										<< std::endl;
					auto begin = std::chrono::high_resolution_clock::now();
					tree_arrangement_engine<float> engine(tc, max_leaf_size, sc);
					bdm.add(
							basic_benchmarker.get_benchmark_data(engine, default_timeout));
					auto end = std::chrono::high_resolution_clock::now();
					std::cerr << "Actual time: "
										<< std::chrono::duration_cast<std::chrono::nanoseconds>(
													 end - begin)
													 .count()
										<< "ns" << std::endl;
					std::cerr << "Completed tree arrangement(tc=" << tc
										<< ",max_leaf_size=" << max_leaf_size << ",sc=" << sc << ")"
										<< std::endl;
				}
			}
		}
		for (size_t tc = 2; tc <= 64; tc *= 2) {
			for (size_t max_leaf_size = 64; max_leaf_size <= 1024 * 4;
					 max_leaf_size *= 4) {
				for (size_t sc = max_leaf_size; sc * tc <= 8192 * 8; sc *= 16 * 2) {
					std::cerr << "Starting tree arrangement_if(tc=" << tc
										<< ",max_leaf_size=" << max_leaf_size << ",sc=" << sc << ")"
										<< std::endl;
					std::cerr << "Expected time proportional to: " << sc * tc
										<< std::endl;
					auto begin = std::chrono::high_resolution_clock::now();
					tree_arrangement_engine_if<float> engine(tc, max_leaf_size, sc);
					bdm.add(
							basic_benchmarker.get_benchmark_data(engine, default_timeout));
					auto end = std::chrono::high_resolution_clock::now();
					std::cerr << "Actual time: "
										<< std::chrono::duration_cast<std::chrono::nanoseconds>(
													 end - begin)
													 .count()
										<< "ns" << std::endl;
					std::cerr << "Completed tree arrangement_if(tc=" << tc
										<< ",max_leaf_size=" << max_leaf_size << ",sc=" << sc << ")"
										<< std::endl;
				}
			}
		}
	}
	return bdm;
}
