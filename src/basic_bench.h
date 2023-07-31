#pragma once

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
		double avg_dist = 0, avg_dist2 = 0;
		for (const auto& q : query_vecs) {
			const auto& ans = eng.query(q);
			T d = dist(q, ans), d2 = dist2(q, ans);
			avg_dist += d;
			avg_dist2 += d2;
		}
		avg_dist /= query_vecs.size();
		avg_dist2 /= query_vecs.size();

		std::cout << "Benchmarking " << eng.name() << '\n';
		std::cout << "\taverage distance: " << avg_dist << '\n';
		std::cout << "\taverage squared distance: " << avg_dist2 << std::endl;
	}
};

template <typename T> void basic_bench<T>::gen_dataset() {
	vec_generator<T> vg;
	for (int i = 0; i < 100; ++i) {
		dataset.push_back(vg.random_vec());
	}
	for (int i = 0; i < 10; ++i) {
		query_vecs.push_back(vg.random_vec());
	}
}

template <typename T> basic_bench<T>::basic_bench() { gen_dataset(); }
