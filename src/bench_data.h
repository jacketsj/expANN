#pragma once

#include <nlohmann/json.hpp>
#include <string>

struct bench_data {
	double time_per_query_ns;
	double time_to_build_ns;
	double average_distance;
	double average_squared_distance;
	double recall;
	std::string engine_name;
};

void to_json(nlohmann::json& j, bench_data bd) {
	j = nlohmann::json{{"time_per_query_ns", bd.time_per_query_ns},
										 {"time_to_build_ns", bd.time_to_build_ns},
										 {"average_distance", bd.average_distance},
										 {"average_squared_distance", bd.average_squared_distance},
										 {"recall", bd.recall},
										 {"engine_name", bd.engine_name}};
}
void from_json(const nlohmann::json& j, bench_data& bd) {
	bd.time_per_query_ns = j.at("time_per_query_ns");
	bd.time_to_build_ns = j.at("time_to_build_ns");
	bd.average_distance = j.at("average_distance");
	bd.average_squared_distance = j.at("average_squared_distance");
	bd.recall = j.at("recall");
	bd.engine_name = j.at("engine_name");
}
