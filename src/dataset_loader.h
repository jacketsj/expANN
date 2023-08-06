#pragma once

#include <vector>

#include "brute_force_engine.h"
#include "dataset.h"
#include "in_memory_dataset.h"

template <typename T> struct dataset_loader {
	in_memory_test_dataset<T>
	load_synethetic_uniform_sphere_points_no_cache(std::string name, size_t n,
																								 size_t m, size_t k, size_t d) {
		in_memory_test_dataset<T> imtd;
		imtd.name = name;
		imtd.n = n;
		imtd.m = m;
		imtd.k = k;
		imtd.dim = d;
		vec_generator<T> vg(d);
		for (size_t i = 0; i < n; ++i) {
			imtd.all_vecs.push_back(vg.random_vec());
		}
		for (size_t i = 0; i < m; ++i) {
			imtd.all_query_vecs.push_back(vg.random_vec());
		}

		brute_force_engine<T> eng;
		for (size_t i = 0; i < imtd.n; ++i)
			eng.store_vector(imtd.get_vec(i));
		eng.build();

		std::cerr << "About to run brute force to get best solutions." << std::endl;
		for (const auto& q : imtd.all_query_vecs) {
			const auto& ans = eng._query_k(q, k);
			imtd.all_query_ans.push_back(ans);
		}
		std::cerr << "Finished running brute force." << std::endl;
		return imtd;
	}
	in_memory_test_dataset<T> load_imtd(std::string filename) {
		std::cerr << "About to load dataset " << filename << std::endl;
		std::ifstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Error opening file: " << filename << std::endl;
			return json();
		}
		json data;
		try {
			file >> data;
		} catch (const std::exception& e) {
			std::cerr << "Error parsing JSON data: " << e.what() << std::endl;
			data = json();
		}
		file.close();
		std::cerr << "Loaded dataset " << filename << std::endl;
		return data.get<in_memory_test_dataset<T>>();
	}
	bool exists_imtd(std::string filename) {
		std::cerr << "Checking if dataset " << filename << " exists" << std::endl;
		std::ifstream file(filename);
		return file.good();
	}
	void save_imtd(std::string filename, const in_memory_test_dataset<T>& imtd) {
		std::cerr << "Saving dataset " << filename << std::endl;
		json data = imtd;
		std::ofstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Error opening file: " << filename << std::endl;
			return;
		}
		file << data.dump(4);
		file.close();
		std::cerr << "Saved dataset " << filename << std::endl;
	}
	in_memory_test_dataset<T> load_synethetic_uniform_sphere_points(size_t n,
																																	size_t m,
																																	size_t k,
																																	size_t d) {
		std::string name = "synthetic_uniform_sphere_n" + std::to_string(n) +
											 "_dim" + std::to_string(d) + "_m" + std::to_string(m) +
											 "_k" + std::to_string(k);
		std::string filename = "./data/" + name + ".dataset";

		// load the file if it exists, else make it
		if (exists_imtd(filename))
			return load_imtd(filename);

		auto imtd =
				load_synethetic_uniform_sphere_points_no_cache(name, n, m, k, d);
		save_imtd(filename, imtd);
		return imtd;
	}
};
