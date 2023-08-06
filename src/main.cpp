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

int main() {
	dataset_loader<float> dsl;
	for (size_t n = 50000; n <= 50000 * 1; n *= 10) {
		size_t m = 400 * (n / 50000);
		size_t d = 16;
		auto bdm = perform_benchmarks(
				dsl.load_synethetic_uniform_sphere_points(n, m, 1, d));
		auto ds_name = bdm.dataset_name;
		std::string data_prefix = "./data/" + ds_name + "/";
		bdm.save(data_prefix);
		std::string plot_prefix = "./plots/" + ds_name + "/";
		make_plots(bdm.get_latest(data_prefix), plot_prefix + "latest_");
		make_plots(bdm.get_all(data_prefix), plot_prefix + "/all_");
	}
}
