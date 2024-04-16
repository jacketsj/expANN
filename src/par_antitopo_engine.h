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

struct par_antitopo_engine_config {
	size_t M;
	size_t ef_construction;
	size_t build_threads;
	size_t ef_search_mult;
	par_antitopo_engine_config(size_t _M, size_t _ef_construction,
														 size_t _build_threads, size_t _ef_search_mult)
			: M(_M), ef_construction(_ef_construction), build_threads(_build_threads),
				ef_search_mult(_ef_search_mult) {}
};

struct par_antitopo_engine : public ann_engine<float, par_antitopo_engine> {
	using fvec = Eigen::VectorXf;
	using dist_t = float;
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_vertex;
	size_t M;
	size_t ef_construction;
	size_t num_threads;
	size_t ef_search_mult;
	size_t max_layer;
#ifdef RECORD_STATS
	size_t num_distcomps = 0;
#endif
	par_antitopo_engine(par_antitopo_engine_config conf)
			: rd(), gen(0), distribution(0, 1), M(conf.M),
				ef_construction(conf.ef_construction), num_threads(conf.build_threads),
				ef_search_mult(conf.ef_search_mult), max_layer(0) {
		// std::cout << "Inside constructor as expected" << std::endl;
	}
	using config = par_antitopo_engine_config;
	std::vector<fvec> all_entries;
	std::vector<fvec> all_entries_extended;
	std::vector<std::vector<std::vector<size_t>>>
			hadj_flat; // vector -> layer -> edges
	std::vector<std::vector<size_t>>
			hadj_bottom; // vector -> edges in bottom layer
	std::vector<std::vector<std::vector<std::pair<dist_t, size_t>>>>
			hadj_flat_with_lengths; // vector -> layer -> edges with lengths
	std::vector<std::vector<std::unique_ptr<std::mutex>>> edge_list_locks;
	void store_vector(const fvec& v);
	fvec to_fvec(const vec<float>& v0) {
		fvec ret = v0.internal;
		return ret;
	}
	void _store_vector(const vec<float>& v0) { store_vector(to_fvec(v0)); }
	void edit_vector(size_t data_index, const fvec& v);
	void improve_entries(const std::vector<size_t>& data_indices);
	std::vector<std::vector<std::pair<dist_t, size_t>>>
	get_knn_per_layer(size_t data_index, std::optional<size_t> thread_index);
	void
	add_edges_with_lock(size_t layer, size_t data_index,
											const std::vector<std::pair<dist_t, size_t>>& new_edges,
											std::barrier<>& mem_barrier);
	void build();
	void _build();
	struct VisitedContainer {
		bool taken;
		std::vector<char> visited; // booleans
		std::vector<size_t> visited_recent;
		VisitedContainer(size_t data_size)
				: taken(false), visited(data_size), visited_recent() {}
	};
	std::vector<std::unique_ptr<VisitedContainer>> visited_containers;
	std::mutex visited_containers_mutex;
	struct VisitedContainerRef {
		std::vector<std::unique_ptr<VisitedContainer>>& visited_containers_ref;
		std::mutex& visited_containers_mutex;
		size_t index;
		bool thread_safe;
		VisitedContainer& vref;
		VisitedContainerRef(
				std::vector<std::unique_ptr<VisitedContainer>>& visited_containers_ref,
				std::mutex& visited_containers_mutex, size_t index, size_t data_size,
				bool thread_safe = false)
				: visited_containers_ref(visited_containers_ref),
					visited_containers_mutex(visited_containers_mutex), index(index),
					thread_safe(thread_safe), vref(*visited_containers_ref[index]) {
			if (vref.taken == true) {
				throw std::runtime_error(
						"Overlapping refs detected! Not thread safe! index=" +
						std::to_string(index));
			}
			if (vref.visited.size() < data_size) {
				vref.visited.resize(data_size);
			}
			vref.taken = true;
		}
		inline bool Visit(const size_t& data_index) {
			if (vref.visited[data_index]) {
				return false;
			}
			vref.visited[data_index] = true;
			vref.visited_recent.emplace_back(data_index);
			return true;
		}
		~VisitedContainerRef() {
			for (const size_t& data_index : vref.visited_recent) {
				vref.visited[data_index] = false;
			}
			vref.visited_recent.clear();
			if (!thread_safe) {
				std::lock_guard<std::mutex> lock(visited_containers_mutex);
				vref.taken = false;
			} else {
				vref.taken = false;
			}
		}
	};
	void resize_visit_containers(size_t num_threads_lower_bound) {
		size_t data_size = all_entries.size();
		if (visited_containers.size() < num_threads_lower_bound) {
			std::lock_guard<std::mutex> lock(visited_containers_mutex);
			while (visited_containers.size() < num_threads_lower_bound) {
				visited_containers.emplace_back(
						std::make_unique<VisitedContainer>(data_size));
			}
		}
	}
	VisitedContainerRef get_visitref(std::optional<size_t> thread_index) {
		if (thread_index.has_value()) {
			resize_visit_containers(thread_index.value() + 1);
			size_t data_size = all_entries.size();
			return VisitedContainerRef(visited_containers, visited_containers_mutex,
																 thread_index.value(), data_size, true);
		}
		std::lock_guard<std::mutex> lock(visited_containers_mutex);
		size_t data_size = all_entries.size();
		size_t index = 0;
		for (; index < visited_containers.size(); ++index) {
			if (!visited_containers[index]->taken) {
				break;
			}
		}
		if (index == visited_containers.size()) {
			visited_containers.emplace_back(
					std::make_unique<VisitedContainer>(data_size));
		}
		return VisitedContainerRef(visited_containers, visited_containers_mutex,
															 index, data_size);
	}
	void update_edges(size_t layer, size_t from) {
		hadj_flat[from][layer].clear();
		hadj_flat[from][layer].reserve(hadj_flat_with_lengths[from][layer].size());
		for (auto& [_, val] : hadj_flat_with_lengths[from][layer]) {
			hadj_flat[from][layer].emplace_back(val);
		}
		if (layer == 0)
			hadj_bottom[from] = hadj_flat[from][layer];
	}
	void add_new_edges(size_t layer, size_t from) {
		for (size_t i = hadj_flat[from][layer].size();
				 i < hadj_flat_with_lengths[from][layer].size(); ++i) {
			hadj_flat[from][layer].emplace_back(
					hadj_flat_with_lengths[from][layer][i].second);
			if (layer == 0)
				hadj_bottom[from].emplace_back(
						hadj_flat_with_lengths[from][layer][i].second);
		}
	}
	void prune_edges(size_t layer, size_t from, bool lazy);
	template <bool use_bottomlayer>
	std::vector<std::pair<dist_t, size_t>>
	query_k_at_layer(const fvec& q, size_t layer,
									 const std::vector<size_t>& entry_points, size_t k,
									 std::optional<size_t> thread_index);
	std::vector<size_t>
	query_k(const fvec& v, size_t k,
					std::optional<size_t> thread_index = std::nullopt);
	std::vector<size_t> _query_k(const vec<float>& v, size_t k);
	const std::string _name() { return "Parallel Anti-Topo Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, ef_search_mult);
		add_param(pl, ef_construction);
#ifdef RECORD_STATS
		add_param(pl, num_distcomps);
#endif
		return pl;
	}
};
