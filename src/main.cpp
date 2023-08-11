#include <iostream>
#include <string>

#include "basic_bench.h"
#include "bench_runner.h"
#include "dataset_loader.h"
#include "matplotlibcpp.h"
#include "plotter.h"
#include "topk_t.h"
#include "vec.h"

#define NUM_THREADS 6

int main() {
	dataset_loader<float> dsl;
	if (true) {
		auto bdm = perform_benchmarks(
				dsl.load_sift1m_easy("datasets/sift/sift_base.fvecs",
														 "datasets/sift/sift_query.fvecs",
														 "datasets/sift/sift_groundtruth.ivecs"),
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
	for (size_t n = 300 * 1; n <= 300 * 10 * 1; n *= 10) {
		// size_t m = 400 * (n / 50000);
		// size_t m = 400;
		size_t m = 300;
		// if (n < 500000)
		//	m = 400;
		size_t d = 16;
		auto bdm = perform_benchmarks(
				dsl.load_synethetic_uniform_sphere_points(n, m, 1, d), NUM_THREADS);
		auto ds_name = bdm.dataset_name;
		std::string data_prefix = "./data/" + ds_name + "/";
		bdm.save(data_prefix);
		std::string plot_prefix = "./plots/" + ds_name + "/";
		make_plots(bdm.get_latest(data_prefix), plot_prefix + "latest_");
		make_plots(bdm.get_all(data_prefix), plot_prefix + "/all_");
	}
}
