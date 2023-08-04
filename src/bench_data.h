#pragma once

#include <string>

struct bench_data {
	double time_per_query_ns;
	double time_to_build_ns;
	double average_distance;
	double average_squared_distance;
	double recall;
	std::string engine_name;
};
