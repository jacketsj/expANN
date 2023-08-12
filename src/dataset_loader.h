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
		imtd.k_want = k;
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
		// for (const auto& q : imtd.all_query_vecs) {
		for (size_t i = 0; i < m; ++i) {
			const auto& ans = eng._query_k(imtd.all_query_vecs[i], k);
			imtd.all_query_ans.push_back(ans);
			assert(imtd.all_query_ans[i].size());
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

		std::cerr << "Dataset did not exists on disk. Creating it." << std::endl;
		auto imtd =
				load_synethetic_uniform_sphere_points_no_cache(name, n, m, k, d);
		save_imtd(filename, imtd);
		return imtd;
	}
	template <typename T2>
	std::vector<std::vector<T2>> Tvecs_read(const std::string& filename) {
		std::ifstream file(filename, std::ios::binary);
		if (!file.is_open()) {
			throw std::runtime_error("I/O error: Unable to open the file " +
															 filename);
		}

		int d;
		file.read(reinterpret_cast<char*>(&d), sizeof(int));

		int vecsizeof = 1 * 4 + d * 4;
		file.seekg(0, std::ios::end);
		int n = file.tellg() / vecsizeof;

		file.seekg(0, std::ios::beg);

		std::vector<T2> data((d + 1) * n);
		file.read(reinterpret_cast<char*>(data.data()), sizeof(T2) * (d + 1) * n);

		std::vector<std::vector<T2>> vectors(n, std::vector<T2>(d));
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < d; ++j) {
				vectors[i][j] = data[(i * (d + 1)) + j + 1];
			}
		}

		file.close();
		return vectors;
	}

	in_memory_test_dataset<float> load_sift1m(std::string filename_base,
																						std::string filename_query,
																						std::string filename_groundtruth) {
		// fvecs format: (d=int,components=float*d) (4+d*4 bytes)
		// ivecs format: (d=int,components=int*d) (4+d*4 bytes)
		// read filename_base as fvecs
		auto base = Tvecs_read<float>(filename_base);
		// read filename_query as fvecs
		auto query = Tvecs_read<float>(filename_query);
		// read filename_groundtruth as ivecs
		auto groundtruth = Tvecs_read<int>(filename_groundtruth);

		in_memory_test_dataset<float> imtd;
		// call imtd.all_vecs.push_back(v)
		for (auto& v : base)
			imtd.all_vecs.emplace_back(v);
		// call imtd.all_query_vecs.push_back(v)
		for (auto& v : base)
			imtd.all_query_vecs.emplace_back(v);
		// TODO call imtd.all_query_ans.push_back({i0,i1,i2,...})
		for (auto& v : groundtruth) {
			imtd.all_query_ans.emplace_back();
			auto& vimtd = imtd.all_query_ans.back();
			for (auto& val : v)
				vimtd.emplace_back(val);
		}

		imtd.name = "sift1m_full_k100";
		imtd.n = base.size();
		imtd.m = query.size();
		imtd.k = groundtruth[0].size();
		imtd.k_want = groundtruth[0].size();
		imtd.dim = base[0].size();

		std::cerr << "Finished loading sift1m. n=" << imtd.n << ",m=" << imtd.m
							<< ",k=" << imtd.k << ",dim=" << imtd.dim << std::endl;

		return imtd;
	}
	in_memory_test_dataset<float>
	load_sift1m_1nn(std::string filename_base, std::string filename_query,
									std::string filename_groundtruth) {
		auto imtd =
				load_sift1m(filename_base, filename_query, filename_groundtruth);
		imtd.k = 1;
		imtd.k_want = 1;
		for (auto& v : imtd.all_query_ans)
			v.resize(imtd.k);
		imtd.name = "sift1m_full";
		return imtd;
	}
	in_memory_test_dataset<float>
	load_sift1m_easy(std::string filename_base, std::string filename_query,
									 std::string filename_groundtruth) {
		auto imtd =
				load_sift1m(filename_base, filename_query, filename_groundtruth);
		imtd.k = 20;
		imtd.k_want = 10;
		imtd.name = "sift1m_easy";
		for (auto& v : imtd.all_query_ans)
			v.resize(imtd.k);
		return imtd;
	}
};
