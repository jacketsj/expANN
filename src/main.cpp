#include <iostream>
#include <string>

#include "arrangement_engine.h"
#include "basic_bench.h"
#include "bench_runner.h"
#include "brute_force_engine.h"
#include "hnsw_engine.h"
#include "matplotlibcpp.h"
#include "plotter.h"
#include "randomgeometry.h"
#include "vec.h"

int main() {
	for (size_t n = 50000; n <= 50000 * 100; n *= 10) {
		size_t m = 400 * (n / 50000);
		auto bdm = perform_benchmarks(n, m);
		std::string size_name = "n" + std::to_string(n) + "_m" + std::to_string(m);
		std::string data_prefix = "./data/" + size_name + "/";
		bdm.save(data_prefix);
		std::string plot_prefix = "./plots/" + size_name + "/";
		make_plots(bdm.get_latest(data_prefix), plot_prefix + "latest_");
		make_plots(bdm.get_all(data_prefix), plot_prefix + "/all_");
	}
}
