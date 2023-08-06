#pragma once

#include <vector>

#include "brute_force_engine.h"
#include "dataset.h"
#include "in_memory_dataset.h"

template <typename T> struct dataset_loader {
	in_memory_test_dataset<T>
	load_synethetic_uniform_sphere_points_no_cache(size_t n, size_t m, size_t k,
																								 size_t d) {
		in_memory_test_dataset<T> imtd;
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

		std::cerr << "About to run brute force to get best solutions (TODO: "
								 "IMPLEMENT CACHING)."
							<< std::endl;
		for (const auto& q : imtd.all_query_vecs) {
			const auto& ans = eng._query_k(q, k);
			imtd.all_query_ans.push_back(ans);
		}
		std::cerr << "Finished running brute force." << std::endl;
		return imtd;
	}
	in_memory_test_dataset<T> load_synethetic_uniform_sphere_points(size_t n,
																																	size_t m,
																																	size_t k,
																																	size_t d) {
		// TODO implement caching
		return load_synethetic_uniform_sphere_points_no_cache(n, m, k, d);
	}
};
// TODO allow generating of synthetic datasets under a name. Generate them
// lazily given a particular name and generator. Name should probably involve
// sizes+dim
