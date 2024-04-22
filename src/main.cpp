#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#define RECORD_STATS 1

#define DIM 128
#include "vec.h"

#include "basic_bench.h"
#include "bench_runner.h"
#include "brute_force_engine.h"
#include "dataset_loader.h"
#include "randomgeometry.h"

using json = nlohmann::json;

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

template <typename T>
T getParameter(int argc, char* argv[], const json& config,
							 const std::string& paramName, const std::string& prompt) {
	if (commandLineOptionExists(argv, argv + argc, ("--" + paramName).c_str())) {
		if constexpr (std::is_same<T, std::string>::value) {
			return getCommandLineOption(argv, argv + argc, "--" + paramName);
		} else {
			return std::stoi(
					getCommandLineOption(argv, argv + argc, "--" + paramName));
		}
	} else if (config.contains(paramName)) {
		return config[paramName].get<T>();
	} else {
		std::cout << prompt;
		T value;
		std::cin >> value;
		return value;
	}
}

int main(int argc, char* argv[]) {
	std::string configFileName = "config.json";
	if (commandLineOptionExists(argv, argv + argc, "--config"))
		configFileName = getCommandLineOption(argv, argv + argc, "--config");
	std::ifstream configFile(configFileName);
	json config;
	if (configFile) {
		configFile >> config;
	}

	std::string dataset = getParameter<std::string>(
			argc, argv, config, "dataset", "Enter dataset type (Synthetic/Sift1M): ");
	std::string ds_name = getParameter<std::string>(argc, argv, config, "ds_name",
																									"Enter dataset name: ");
	int num_threads = getParameter<int>(argc, argv, config, "num_threads",
																			"Enter number of threads: ");

	dataset_loader<float> dsl;
	std::optional<bench_data_manager> bdm;

	if (dataset == "Sift1M") {
		size_t k = getParameter<size_t>(argc, argv, config, "k",
																		"Enter Sift1M dataset parameter k: ");
		std::cout << "Using Sift1M dataset with k=" << k << std::endl;
		bdm = perform_benchmarks(
				dsl.load_sift1m("datasets/sift/sift_base.fvecs",
												"datasets/sift/sift_query.fvecs",
												"datasets/sift/sift_groundtruth.ivecs", k),
				num_threads);
	} else if (dataset == "Synthetic") {
		size_t n = getParameter<size_t>(argc, argv, config, "n",
																		"Enter Synthetic dataset parameter n: ");
		size_t m = getParameter<size_t>(argc, argv, config, "m",
																		"Enter Synthetic dataset parameter m: ");
		size_t d = getParameter<size_t>(argc, argv, config, "d",
																		"Enter Synthetic dataset parameter d: ");
		size_t k = getParameter<size_t>(argc, argv, config, "k",
																		"Enter Synthetic dataset parameter k: ");

		std::cout << "Using Synthetic dataset with n,m,d,k=" << n << ',' << m << ','
							<< d << ',' << k << std::endl;
		bdm = perform_benchmarks(
				dsl.load_synethetic_uniform_sphere_points(n, m, k, d), num_threads);
	} else {
		std::cerr << "Invalid dataset type!" << std::endl;
		return 1;
	}

	if (ds_name.empty()) {
		ds_name =
				dataset; // Default dataset name to the type of dataset if not specified
	}

	std::string data_prefix = "./data/" + ds_name + "/";
	bdm->save(data_prefix);

	return 0;
}
