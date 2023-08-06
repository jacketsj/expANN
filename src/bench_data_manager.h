#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <variant>
#include <vector>

#include "bench_data.h"

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

	// Ensure the directories exist before writing the file
	std::filesystem::create_directories(
			std::filesystem::path(filename).parent_path());

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
	void save(std::string prefix = "") {
		serialize_vector_to_json(latest, prefix + bd_latest_filename, false);
		serialize_vector_to_json(latest, prefix + bd_all_filename, true);
	}
	std::vector<bench_data> get_latest(std::string _prefix = "") {
		auto copy = latest;
		// latest.clear();
		return copy;
	}
	std::vector<bench_data> get_all(std::string prefix = "") {
		return deserialize_json_to_vector<bench_data>(prefix + bd_all_filename);
	}
	void add(std::variant<bench_data, std::string> bdv) {
		if (std::holds_alternative<bench_data>(bdv))
			latest.push_back(std::get<bench_data>(bdv));
		else
			std::cerr << "Got bench error: " << std::get<std::string>(bdv)
								<< std::endl;
	}
};
