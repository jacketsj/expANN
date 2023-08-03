#pragma once

#include <map>
#include <string>
#include <vector>

#include "basic_bench.h"
#include "matplotlibcpp.h"

#include "arrangement_engine.h"
#include "brute_force_engine.h"
#include "hnsw_engine.h"
#include "hnsw_engine_2.h"

namespace plt = matplotlibcpp;

void make_plots(const std::vector<bench_data>& dataset) {
	std::map<std::string, std::vector<bench_data>> entries_by_name;
	for (const auto& bd : dataset) {
		entries_by_name[bd.engine_name].push_back(bd);
	}

	double pointsize = 10.0;

	// querytime-recall
	plt::figure_size(1280, 720);
	for (const auto& [name, bdv] : entries_by_name) {
		std::vector<double> tpq, recall;
		for (const auto& bd : bdv) {
			tpq.push_back(bd.time_per_query_ns);
			recall.push_back(bd.recall);
		}
		// plt::clear();
		plt::scatter(recall, tpq, pointsize, {{"label", name}});
		// plt::title(name + "_recall-querytime");
		// plt::save("./plots/" + name + ".png");
	}
	plt::xlabel("recall");
	plt::ylabel("querytime(ns)");
	// plt::xscale("log");
	plt::yscale("log");
	plt::legend();
	plt::title("recall-querytime for 1-NN");
	plt::save("./plots/time-recall.png");

	// querytime-avgdist
	plt::figure_size(1280, 720);
	for (const auto& [name, bdv] : entries_by_name) {
		std::vector<double> tpq, avgdist;
		for (const auto& bd : bdv) {
			tpq.push_back(bd.time_per_query_ns);
			avgdist.push_back(bd.average_distance);
		}
		plt::scatter(avgdist, tpq, pointsize, {{"label", name}});
	}
	plt::xlabel("avgdist");
	plt::ylabel("querytime(ns)");
	// plt::xscale("log");
	plt::yscale("log");
	plt::legend();
	plt::title("avgdist-querytime for 1-NN");
	plt::save("./plots/time-avgdist.png");

	// buildtime-recall
	plt::figure_size(1280, 720);
	for (const auto& [name, bdv] : entries_by_name) {
		std::vector<double> y, x;
		for (const auto& bd : bdv) {
			x.push_back(bd.time_to_build_ns);
			y.push_back(bd.recall);
		}
		plt::scatter(x, y, pointsize, {{"label", name}});
	}
	plt::xlabel("time_to_build(ns)");
	plt::ylabel("recall");
	plt::xscale("log");
	plt::yscale("log");
	plt::legend();
	plt::title("timetobuild-recall for 1-NN");
	plt::save("./plots/buildtime-recall.png");
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

	//{
	//	hnsw_engine_2<float> engine2(100, 60);
	//	benchmark_dataset.push_back(basic_benchmarker.get_benchmark_data(engine2));
	//	std::cerr << "Completed hnsw2(60,0.5)" << std::endl;
	//}

	for (size_t k = 100; k <= 140; k += 40) {
		for (int p2 = 15; p2 < 18; ++p2) {
			if (p2 > 4)
				p2 += 2;
			std::cerr << "About to start hnsw(k=" << k << ",p2=" << p2 << ")"
								<< std::endl;
			hnsw_engine<float, false> engine(50, k, 0.5 * p2);
			benchmark_dataset.push_back(basic_benchmarker.get_benchmark_data(engine));
			std::cerr << "Completed hnsw(k=" << k << ",p2=" << p2 << ")" << std::endl;
		}
	}
	for (size_t k = 2; k <= 60; k += 12) {
		for (size_t num_for_1nn = 1; num_for_1nn <= 40; num_for_1nn *= 4) {
			std::cerr << "About to start hnsw2(k=" << k << ",n4nn=" << num_for_1nn
								<< ")" << std::endl;
			hnsw_engine_2<float> engine2(100, k, num_for_1nn);
			benchmark_dataset.push_back(
					basic_benchmarker.get_benchmark_data(engine2));
			std::cerr << "Completed hnsw2(k=" << k << ")" << std::endl;
		}
	}

	// hnsw_engine<float, false> big_hnsw_engine(200, 200, 15);
	// benchmark_dataset.push_back(
	//		basic_benchmarker.get_benchmark_data(big_hnsw_engine));
	// std::cerr << "Completed big hnsw" << std::endl;

	for (size_t k = 1; k < 4; ++k) {
		for (size_t n = 4; n <= 256; n *= 2) {
			for (size_t m = 1; m <= 64; m *= 4) {
				arrangement_engine<float> engine(k, n, m);
				benchmark_dataset.push_back(
						basic_benchmarker.get_benchmark_data(engine));
				std::cerr << "Completed arrangement(k=" << k << ",n=" << n << ",m=" << m
									<< ")" << std::endl;
				// use_engine(engine_arrange);
			}
		}
	}

	return benchmark_dataset;
}
