#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "basic_bench.h"
#include "bench_data_manager.h"

#include "brute_force_engine.h"
#include "dataset.h"
#include "dataset_loader.h"
#include "ehnsw_engine_2.h"
#include "ehnsw_engine_3.h"
#include "hnsw_engine_2.h"
#include "hnsw_engine_3.h"
#include "tree_arrangement_engine.h"
#include "tree_arrangement_engine_if.h"

struct job {
	std::function<void()> f;
	std::string name;
	param_list_t param_list;
	job(std::function<void()> _f, std::string _name, param_list_t _param_list)
			: f(_f), name(_name), param_list(_param_list) {}
	std ::string to_string() const {
		std::string ret;
		ret += name;
		ret += "(";
		for (const auto& [k, v] : param_list) {
			ret += k + '=' + v + ',';
		}
		ret[ret.size() - 1] = ')';
		return ret;
	}
	void run(size_t t, size_t job_index, size_t num_jobs) {
		std::printf("Running job (t=%zu, job %zu/%zu): %s\n", t, job_index,
								num_jobs, to_string().c_str());
		// std::cerr << "Running job (t=" << t << "): " << to_string() << std::endl;
		f();
		// std::cerr << "Completed job (t=" << t << "): " << to_string() <<
		// std::endl;
		std::printf("Completed job (t=%zu, job %zu/%zu): %s\n", t, job_index,
								num_jobs, to_string().c_str());
	}
};

template <typename test_dataset_t>
bench_data_manager perform_benchmarks(test_dataset_t ds, size_t num_threads) {
	basic_bench<float, test_dataset_t> basic_benchmarker(ds);
	bench_data_manager bdm(ds.name);

	// TODO implement a way to disable the timeout in get_benchmark_data (e.g. if
	// it's 0, or at least a way to make it super long)
	std::cerr << "Running benchmarks with a 6000s timeout" << std::endl;

	using namespace std::chrono_literals;
	auto default_timeout = 6000s;

	std::vector<job> jobs;
	std::vector<std::variant<bench_data, std::string>> job_results;

	auto add_engine = [&jobs, &job_results, &basic_benchmarker,
										 &default_timeout](auto engine_gen) mutable {
		size_t i = jobs.size();
		job_results.emplace_back();
		jobs.emplace_back(
				[&basic_benchmarker, i, engine_gen, default_timeout,
				 &job_results]() mutable {
					auto engine = engine_gen();
					job_results[i] =
							basic_benchmarker.get_benchmark_data(engine, default_timeout);
				},
				engine_gen().name(), engine_gen().param_list());
	};

	for (const auto& bd : job_results)
		bdm.add(bd);

	auto add_hnsw2 = [&](size_t max_depth, size_t k, size_t num_for_1nn) {
		auto engine_gen = [&] {
			return hnsw_engine_2<float>(max_depth, k, num_for_1nn);
		};
		// add_engine(engine_gen);
	};

	auto add_hnsw3 = [&](size_t max_depth, size_t k, size_t num_for_1nn) {
		auto engine_gen = [&] {
			return hnsw_engine_3<float>(max_depth, k, num_for_1nn);
		};
		// add_engine(engine_gen);
	};

	// brute_force_engine<float> engine_bf;
	// bdm.add(basic_benchmarker.get_benchmark_data(engine_bf, default_timeout));

	// for (size_t k = 38; k <= 50; k += 10) {
	for (size_t k = 20; k <= 80; k += 10) {
		for (size_t num_for_1nn = 32 * 4; num_for_1nn <= 32 * 8; num_for_1nn *= 2) {
			add_hnsw2(100, k, num_for_1nn);
			add_hnsw3(100, k, num_for_1nn);
		}
	}
	auto add_ehnsw2 = [&](size_t edge_count_mult, size_t max_depth,
												size_t min_per_cut, size_t num_cuts,
												size_t num_for_1nn) {
		auto engine_gen = [&] {
			return ehnsw_engine_2<float>(max_depth, edge_count_mult, num_for_1nn,
																	 num_cuts, min_per_cut);
		};
		add_engine(engine_gen);
	};
	auto add_ehnsw3 = [&](size_t edge_count_mult, size_t max_depth,
												size_t min_per_cut, size_t num_cuts,
												size_t num_for_1nn) {
		auto engine_gen = [&] {
			return ehnsw_engine_3<float>(max_depth, edge_count_mult, num_for_1nn,
																	 num_cuts, min_per_cut);
		};
		// add_engine(engine_gen);
	};
	//	for (size_t ecm = 10; ecm <= 40; ecm *= 2)
	//		for (size_t mpc = 1; mpc <= 8; mpc *= 2)
	//			for (size_t nc = 1; nc <= 8; nc *= 2)
	//				for (size_t n4nn = 1; n4nn <= 16; n4nn *= 4)
	// for (size_t ecm = 2; ecm <= 10; ecm += 1)
	for (size_t ecm = 47; ecm <= 47; ecm += 1)
		for (size_t mpc = 1; mpc <= 4; mpc *= 2)
			for (size_t nc = 1; nc * mpc < ecm; nc *= 2)
				for (size_t n4nn = 16; n4nn <= 16; n4nn *= 4) {
					add_ehnsw2(ecm, 100, mpc, nc, n4nn);
					add_ehnsw3(ecm, 100, mpc, nc, n4nn);
				}
	// add_hnsw2(56, 100, 1, 16, 32);
	// add_hnsw2(56, 100, 4, 8, 8);
	// add_hnsw2(47, 100, 1, 16, 64);
	// add_hnsw2(46, 100, 1, 4, 128);

	auto add_tae = [&](size_t tc, size_t max_leaf_size, size_t sc) {
		auto engine_gen = [&] {
			return tree_arrangement_engine<float>(tc, max_leaf_size, sc);
		};
		// add_engine(engine_gen);
	};

	for (size_t tc = 2; tc <= 64; tc *= 2) {
		for (size_t max_leaf_size = 64; max_leaf_size <= 1024 * 4;
				 max_leaf_size *= 4) {
			for (size_t sc = max_leaf_size; sc * tc <= 8192 * 8; sc *= 16 * 2) {
				add_tae(tc, max_leaf_size, sc);
			}
		}
	}

	auto add_tae_if = [&](size_t tc, size_t max_leaf_size, size_t sc) {
		auto engine_gen = [&] {
			return tree_arrangement_engine_if<float>(tc, max_leaf_size, sc);
		};
		// add_engine(engine_gen);
	};

	for (size_t tc = 2; tc <= 64; tc *= 2) {
		for (size_t max_leaf_size = 64; max_leaf_size <= 1024 * 4;
				 max_leaf_size *= 4) {
			for (size_t sc = max_leaf_size; sc * tc <= 8192 * 8; sc *= 16 * 2) {
				add_tae_if(tc, max_leaf_size, sc);
			}
		}
	}

	{
		std::vector<std::jthread> threadpool;
		std::atomic_size_t g_job_index = 0;
		for (size_t t_index = 0; t_index < num_threads; ++t_index) {
			threadpool.emplace_back([&]() {
				for (size_t t_job_index = g_job_index++; t_job_index < jobs.size();
						 t_job_index = g_job_index++) {
					jobs[t_job_index].run(t_index, t_job_index, jobs.size());
				}
			});
		}
	}
	for (size_t job_index = 0; job_index < job_results.size(); ++job_index) {
		bdm.add(job_results[job_index]);
	}
	return bdm;
}
