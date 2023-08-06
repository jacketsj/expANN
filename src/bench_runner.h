#pragma once

#include <string>
#include <vector>

#include "basic_bench.h"
#include "bench_data_manager.h"

#include "arrangement_engine.h"
#include "brute_force_engine.h"
#include "hier_arrangement_engine.h"
#include "hnsw_engine.h"
#include "hnsw_engine_2.h"
#include "hnsw_engine_hybrid.h"
#include "tree_arrangement_engine.h"

bench_data_manager perform_benchmarks(size_t bench_n, size_t bench_m) {
	basic_bench<float> basic_benchmarker(bench_n, bench_m);
	bench_data_manager bdm;

	std::cerr << "About to run brute force to get best solutions." << std::endl;
	brute_force_engine<float> engine_bf;
	basic_benchmarker.populate_ans(engine_bf);
	std::cerr << "Finished running brute force." << std::endl;

	// bdm.add(basic_benchmarker.get_benchmark_data(engine_bf));

	using namespace std::chrono_literals;
	auto default_timeout = 3s;

	for (size_t k = 100; k <= 140; k += 40) {
		for (int p2 = 15; p2 < 18; ++p2) {
			if (p2 > 4)
				p2 += 2;
			std::cerr << "About to start hnsw(k=" << k << ",p2=" << p2 << ")"
								<< std::endl;
			hnsw_engine<float, false> engine(50, k, 0.5 * p2);
			bdm.add(basic_benchmarker.get_benchmark_data(engine, default_timeout));
			std::cerr << "Completed hnsw(k=" << k << ",p2=" << p2 << ")" << std::endl;
		}
	}
	for (size_t k = 2; k <= 60; k += 12) {
		for (size_t num_for_1nn = 1; num_for_1nn <= 40; num_for_1nn *= 4) {
			std::cerr << "About to start hnsw2(k=" << k << ",n4nn=" << num_for_1nn
								<< ")" << std::endl;
			hnsw_engine_2<float> engine2(100, k, num_for_1nn);
			bdm.add(basic_benchmarker.get_benchmark_data(engine2, default_timeout));
			std::cerr << "Completed hnsw2(k=" << k << ")" << std::endl;
		}
	}
	for (size_t k = 2; k <= 60; k += 12) {
		for (size_t num_for_1nn = 1; num_for_1nn <= 40; num_for_1nn *= 4) {
			std::cerr << "About to start hnsw_hybrid(k=" << k
								<< ",n4nn=" << num_for_1nn << ")" << std::endl;
			hnsw_engine_hybrid<float> engine(100, k, num_for_1nn);
			bdm.add(basic_benchmarker.get_benchmark_data(engine, default_timeout));
			std::cerr << "Completed hnsw_hybrid(k=" << k << ",n4nn=" << num_for_1nn
								<< ")" << std::endl;
		}
	}

	for (size_t k = 1; k < 4; ++k) {
		for (size_t n = 4; n <= 256; n *= 2) {
			for (size_t m = 1; m <= 64; m *= 4) {
				std::cerr << "Starting arrangement(k=" << k << ",n=" << n << ",m=" << m
									<< ")" << std::endl;
				arrangement_engine<float> engine(k, n, m);
				bdm.add(basic_benchmarker.get_benchmark_data(engine, default_timeout));
				std::cerr << "Completed arrangement(k=" << k << ",n=" << n << ",m=" << m
									<< ")" << std::endl;
			}
		}

		for (size_t na = 1; na <= 16; na *= 4) {
			for (size_t levels = 1; levels * na <= 32; levels *= 2) {
				for (size_t sc = 32; sc <= 8192; sc *= 4) {
					std::cerr << "Starting hier arrangement(na=" << na
										<< ",levels=" << levels << ",sc=" << sc << ")" << std::endl;
					hier_arrangement_engine<float> engine(na, levels, sc);
					bdm.add(
							basic_benchmarker.get_benchmark_data(engine, default_timeout));
					std::cerr << "Completed hier arrangement(na=" << na
										<< ",levels=" << levels << ",sc=" << sc << ")" << std::endl;
				}
			}
		}
	}
	for (size_t tc = 2; tc <= 14; tc += 3) {
		for (size_t max_leaf_size = 64; max_leaf_size <= 1024 * 8;
				 max_leaf_size *= 16) {
			for (size_t sc = max_leaf_size; sc * tc <= 8192 * 8; sc *= 16 * 2) {
				std::cerr << "Starting tree arrangement(tc=" << tc
									<< ",max_leaf_size=" << max_leaf_size << ",sc=" << sc << ")"
									<< std::endl;
				std::cerr << "Expected time proportional to: " << sc * tc << std::endl;
				auto begin = std::chrono::high_resolution_clock::now();
				tree_arrangement_engine<float> engine(tc, max_leaf_size, sc);
				bdm.add(basic_benchmarker.get_benchmark_data(engine, default_timeout));
				auto end = std::chrono::high_resolution_clock::now();
				std::cerr << "Actual time: "
									<< std::chrono::duration_cast<std::chrono::nanoseconds>(end -
																																					begin)
												 .count()
									<< "ns" << std::endl;
				std::cerr << "Completed tree arrangement(tc=" << tc
									<< ",max_leaf_size=" << max_leaf_size << ",sc=" << sc << ")"
									<< std::endl;
			}
		}
	}
	return bdm;
}
