#pragma once

#include <chrono>
#include <vector>

#include "ann_engine.h"
#include "randomgeometry.h"
#include "vec.h"

template <typename T> struct basic_bench {
	std::vector<vec<T>> dataset;
	std::vector<vec<T>> query_vecs;
	basic_bench();
	void gen_dataset();
	template <class Engine> void perform_benchmark(ann_engine<T, Engine>& eng) {
		// store all vectors in the engine
		for (const auto& v : dataset)
			eng.store_vector(v);
		// build the engine
		eng.build();
		// run the queries
		double avg_dist = 0, avg_dist2 = 0;
		auto time_begin = std::chrono::high_resolution_clock::now();
		for (const auto& q : query_vecs) {
			const auto& ans = eng.query(q);
			T d = dist(q, ans), d2 = dist2(q, ans);
			avg_dist += d;
			avg_dist2 += d2;
		}
		auto time_end = std::chrono::high_resolution_clock::now();
		avg_dist /= query_vecs.size();
		avg_dist2 /= query_vecs.size();

		std::cout << "Benchmarking " << eng.name() << '\n';
		std::cout << "\taverage distance: " << avg_dist << '\n';
		std::cout << "\taverage squared distance: " << avg_dist2 << '\n';
		std::cout << "\taverage query time: "
							<< std::chrono::duration_cast<std::chrono::nanoseconds>(
										 time_end - time_begin)
												 .count() /
										 query_vecs.size()
							<< "ns" << std::endl;
	}
};

template <typename T> void basic_bench<T>::gen_dataset() {
	vec_generator<T> vg;
	for (int i = 0; i < 500; ++i) {
		dataset.push_back(vg.random_vec());
	}
	for (int i = 0; i < 10; ++i) {
		query_vecs.push_back(vg.random_vec());
	}
}

template <typename T> basic_bench<T>::basic_bench() { gen_dataset(); }
