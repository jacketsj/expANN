#pragma once

#include <nlohmann/json.hpp>
#include <string>

#include "ann_engine.h"

struct cluster_bench_data {
	double total_time_ns;
	double score;
	std::string engine_name;
	param_list_t param_list;

	std::string to_string() const;
};

void to_json(nlohmann::json& j, cluster_bench_data bd) {
	j = nlohmann::json{{"total_time_ns", bd.total_time_ns},
										 {"score", bd.score},
										 {"engine_name", bd.engine_name},
										 {"param_list", nlohmann::json(bd.param_list)}};
}
void from_json(const nlohmann::json& j, cluster_bench_data& bd) {
	bd.total_time_ns = j.at("total_time_ns");
	bd.score = j.at("score");
	bd.engine_name = j.at("engine_name");
	if (j.contains("param_list")) {
		bd.param_list = j.at("param_list").get<param_list_t>();
	}
}

std::string cluster_bench_data::to_string() const {
	return nlohmann::json(*this).dump();
}
