#pragma once

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

#include "bench_data.h"

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

using json = nlohmann::json;

// Serialize a vector of any type T to a JSON file, appending to any existing
// files
template <typename T>
void serialize_vector_to_json(const std::vector<T>& data,
															const std::string& filename, bool append) {
	json j;

	// If the file already exists and contains valid data, read and append to it
	if (std::ifstream(filename)) {
		std::ifstream inputFile(filename);
		json existingData;
		inputFile >> existingData;
		if (append)
			j = existingData;
	}

	// Append the new data to the JSON array
	for (const auto& item : data) {
		j.emplace_back(item);
	}

	// Write the updated data to the file
	std::ofstream outputFile(filename);
	outputFile << j.dump(4); // Pretty print with indentation of 4 spaces
}

// Deserialize a JSON file to a vector of any type T
template <typename T>
std::vector<T> deserialize_json_to_vector(const std::string& filename) {
	std::vector<T> data;
	json j;

	std::ifstream inputFile(filename);
	if (inputFile) {
		inputFile >> j;
		inputFile.close();
	}

	// Convert the JSON array to a vector of T
	if (j.is_array()) {
		data = j.get<std::vector<T>>();
	}

	return data;
}

struct bench_data_manager {
	std::string bd_all_filename = "data/all.json";
	std::string bd_latest_filename = "data/latest.json";
	std::vector<bench_data> latest;
	void save() {
		serialize_vector_to_json(latest, bd_latest_filename, false);
		serialize_vector_to_json(latest, bd_all_filename, true);
	}
	std::vector<bench_data> get_latest() {
		auto copy = latest;
		// latest.clear();
		return copy;
	}
	std::vector<bench_data> get_all() {
		return deserialize_json_to_vector<bench_data>(bd_all_filename);
	}
	void add(bench_data bd) { latest.push_back(bd); }
};
