#pragma once

#include <map>
#include <string>
#include <vector>

#include "basic_bench.h"
#include "matplotlibcpp.h"

#include "arrangement_engine.h"
#include "brute_force_engine.h"
#include "hnsw_engine.h"

namespace plt = matplotlibcpp;

void make_plots(const std::vector<bench_data>& dataset) {
	std::map<std::string, std::vector<bench_data>> entries_by_name;
	for (const auto& bd : dataset) {
		entries_by_name[bd.engine_name].push_back(bd);
	}
	plt::figure_size(1280, 720);
	for (const auto& [name, bdv] : entries_by_name) {
		std::vector<double> tpq, recall;
		for (const auto& bd : bdv) {
			tpq.push_back(bd.time_per_query_ns);
			recall.push_back(bd.recall);
		}
		// plt::clear();
		plt::scatter(recall, tpq, 1.0, {{"label", name}});
		// plt::title(name + "_recall-querytime");
		// plt::save("./plots/" + name + ".png");
	}
	plt::xlabel("recall");
	plt::ylabel("querytime(ns)");
	// plt::xscale("log");
	plt::yscale("log");
	plt::legend();
	plt::title("recall-querytime");
	plt::save("./plots/all.png");
}

std::vector<bench_data> perform_benchmarks() {
	basic_bench<float> basic_benchmarker;
	std::vector<bench_data> benchmark_dataset;

	brute_force_engine<float> engine_bf;
	basic_benchmarker.populate_ans(engine_bf);

	// auto use_engine = [&]<typename Eng>(Eng engine) {
	//	benchmark_dataset.push_back(basic_benchmarker.get_benchmark_data(engine));
	// };

	benchmark_dataset.push_back(basic_benchmarker.get_benchmark_data(engine_bf));
	// use_engine(engine_bf);

	for (size_t k = 10; k <= 60; k += 10) {
		for (int p2 = 2; p2 < 12; ++p2) {
			if (p2 > 4)
				p2 += 2;
			// for (int p2 = 1; p2 <= 8; ++p2) {
			hnsw_engine<float, false> engine(50, k, 0.5 * p2);
			benchmark_dataset.push_back(basic_benchmarker.get_benchmark_data(engine));
			std::cerr << "Completed hnsw(k=" << k << ",p2=" << p2 << ")" << std::endl;
			// use_engine(engine_hnsw);
		}
	}
	hnsw_engine<float, false> big_hnsw_engine(100, 40, 15);
	benchmark_dataset.push_back(
			basic_benchmarker.get_benchmark_data(big_hnsw_engine));
	for (size_t k = 1; k < 4; ++k) {
		for (size_t n = 4; n <= 256; n *= 2) {
			arrangement_engine<float> engine(k, n);
			benchmark_dataset.push_back(basic_benchmarker.get_benchmark_data(engine));
			std::cerr << "Completed arrangement(k=" << k << ",n=" << n << ")"
								<< std::endl;
			// use_engine(engine_arrange);
		}
	}

	return benchmark_dataset;
}
