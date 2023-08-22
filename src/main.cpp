#include <iostream>
#include <string>

#include "arrangement_engine.h"
#include "basic_bench.h"
#include "bench_runner.h"
#include "brute_force_engine.h"
#include "dataset_loader.h"
#include "hnsw_engine.h"
#include "matplotlibcpp.h"
#include "plotter.h"
#include "randomgeometry.h"
#include "vec.h"

#define NUM_THREADS 8

int main() {
	dataset_loader<float> dsl;
	if (false) {
		// auto bdm = perform_benchmarks(dsl.load_sift1m_custom(
		//		"datasets/sift/sift_base.fvecs", "datasets/sift/sift_query.fvecs",
		//		"datasets/sift/sift_groundtruth.ivecs", 10, 8));
		auto bdm = perform_benchmarks(
				dsl.load_sift1m("datasets/sift/sift_base.fvecs",
												"datasets/sift/sift_query.fvecs",
												"datasets/sift/sift_groundtruth.ivecs", 10),
				NUM_THREADS);
		auto ds_name = bdm.dataset_name;
		std::string data_prefix = "./data/" + ds_name + "/";
		bdm.save(data_prefix);
		std::string plot_prefix = "./plots/" + ds_name + "/";
		make_plots(bdm.get_latest(data_prefix), plot_prefix + "latest_");
		make_plots(bdm.get_all(data_prefix), plot_prefix + "/all_");

		return 0;
	}

	// for (size_t n = 50000 * 1; n <= 50000 * 10 * 1; n *= 10) {
	for (size_t n = 45000 * 1; n <= 45000 * 1 * 1; n *= 10) {
		// for (size_t n = 10 * 1; n <= 10 * 1 * 1; n *= 10) {
		// size_t m = 400 * (n / 50000);
		// size_t m = 400;
		size_t m = 400;
		// size_t m = 4;
		// if (n < 500000)
		//	m = 400;
		size_t d = 128;
		size_t k = 10;
		// size_t k = 1;
		auto bdm = perform_benchmarks(
				dsl.load_synethetic_uniform_sphere_points(n, m, k, d), NUM_THREADS);
		auto ds_name = bdm.dataset_name;
		std::string data_prefix = "./data/" + ds_name + "/";
		bdm.save(data_prefix);
		std::string plot_prefix = "./plots/" + ds_name + "/";
		make_plots(bdm.get_latest(data_prefix), plot_prefix + "latest_");
		make_plots(bdm.get_all(data_prefix), plot_prefix + "/all_");
	}
}
