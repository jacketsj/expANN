#pragma once

#include <algorithm>
#include <barrier>
#include <iostream>
#include <latch>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "ann_engine.h"
#include "vec.h"

#include "antitopo_engine_basic.h"

struct par_antitopo_engine_config {
	size_t M;
	size_t ef_construction;
	size_t build_threads;
	size_t ef_search_mult;
	bool use_mips;
	par_antitopo_engine_config(size_t _M, size_t _ef_construction,
														 size_t _build_threads, size_t _ef_search_mult,
														 bool _use_mips)
			: M(_M), ef_construction(_ef_construction), build_threads(_build_threads),
				ef_search_mult(_ef_search_mult), use_mips(_use_mips) {}
};

struct par_antitopo_engine : public ann_engine<float, par_antitopo_engine> {
	using fvec = Eigen::VectorXf;
	Mop::Rough::antitopo_engine sub_engine;
	using config = par_antitopo_engine_config;
	config conf;
	par_antitopo_engine(par_antitopo_engine_config conf)
			: sub_engine(Mop::Rough::antitopo_engine_build_config(
						conf.M, conf.ef_construction, conf.build_threads, conf.use_mips)),
				conf(conf) {}
	fvec to_fvec(const vec<float>& v0) {
		fvec ret = v0.internal;
		return ret;
	}
	void _store_vector(const vec<float>& v0) {
		sub_engine.store_vector(to_fvec(v0));
	}
	void _build() { sub_engine.build(); }
	std::vector<size_t> _query_k(const vec<float>& q, size_t k) {
		return sub_engine.query_k(
				to_fvec(q), k,
				Mop::Rough::antitopo_engine_query_config(conf.ef_search_mult));
	}
	const std::string _name() { return "MopBucket Parallel Anti-Topo Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, conf.M);
		add_param(pl, conf.ef_search_mult);
		add_param(pl, conf.ef_construction);
		add_param(pl, conf.build_threads);
		add_param(pl, conf.use_mips);
		return pl;
	}
};
