#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#define RECORD_STATS 1

#define DIM 128
#include "vec.h"

#include "arrangement_engine.h"
#include "basic_bench.h"
#include "bench_runner.h"
#include "brute_force_engine.h"
#include "dataset_loader.h"
#include "hnsw_engine.h"
#include "matplotlibcpp.h"
#include "plotter.h"
#include "randomgeometry.h"

std::string getCommandLineOption(char** begin, char** end,
																 const std::string& option) {
	char** itr = std::find(begin, end, option);
	if (itr != end && ++itr != end) {
		return std::string(*itr);
	}
	return "";
}

bool commandLineOptionExists(char** begin, char** end,
														 const std::string& option) {
	return std::find(begin, end, option) != end;
}

int main(int argc, char* argv[]) {
	std::string dataset;
	std::string ds_name;
	int num_threads = 1; // Default number of threads

	if (commandLineOptionExists(argv, argv + argc, "--dataset")) {
		dataset = getCommandLineOption(argv, argv + argc, "--dataset");
	} else {
		std::cout << "Enter dataset type (Synthetic/Sift1M): ";
		std::cin >> dataset;
	}

	if (commandLineOptionExists(argv, argv + argc, "--ds_name")) {
		ds_name = getCommandLineOption(argv, argv + argc, "--ds_name");
	} else {
		std::cout << "Enter dataset name (leave blank for default): ";
		std::getline(std::cin >> std::ws, ds_name);
		if (ds_name.empty()) {
			ds_name = dataset;
		}
	}

	if (commandLineOptionExists(argv, argv + argc, "--num_threads")) {
		num_threads =
				std::stoi(getCommandLineOption(argv, argv + argc, "--num_threads"));
	}
	std::cout << "Using " << num_threads << " threads." << std::endl;

	dataset_loader<float> dsl;
	std::optional<bench_data_manager> bdm;

	if (dataset == "Sift1M") {
		size_t k = 10; // default value for k
		bdm = perform_benchmarks(
				dsl.load_sift1m("datasets/sift/sift_base.fvecs",
												"datasets/sift/sift_query.fvecs",
												"datasets/sift/sift_groundtruth.ivecs", k),
				num_threads);
	} else if (dataset == "Synthetic") {
		size_t n, m, d, k;
		std::cout << "Enter Synthetic dataset parameters n, m, d, k: ";
		std::cin >> n >> m >> d >> k;
		bdm = perform_benchmarks(
				dsl.load_synethetic_uniform_sphere_points(n, m, k, d), num_threads);
	} else {
		std::cerr << "Invalid dataset type!" << std::endl;
		return 1;
	}

	std::string data_prefix = "./data/" + ds_name + "/";
	bdm->save(data_prefix);
	std::string plot_prefix = "./plots/" + ds_name + "/";
	make_plots(bdm->get_latest(data_prefix), plot_prefix + "latest_");
	make_plots(bdm->get_all(data_prefix), plot_prefix + "/all_");

	return 0;
}
