#include <iostream>

#include "arrangement_engine.h"
#include "basic_bench.h"
#include "brute_force_engine.h"
#include "hnsw_engine.h"
#include "matplotlibcpp.h"
#include "plotter.h"
#include "randomgeometry.h"
#include "vec.h"

int main() {
	auto bdm = perform_benchmarks();
	bdm.save();
	make_plots(bdm.get_latest(), "./plots/latest_");
	make_plots(bdm.get_all(), "./plots/all_");
}
