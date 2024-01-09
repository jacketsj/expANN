#pragma once

#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ann_engine.h"
#include "product_quantizer_3.h"
#include "randomgeometry.h"
#include "robin_hood.h"
#include "topk_t.h"

#include "ehnsw_engine_basic_fast.h"

struct ehnsw_engine_basic_fast_clusterchunks_config {
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	bool use_cuts;
	size_t min_cluster_size;
	size_t max_cluster_size;
	bool very_early_termination;
	bool use_clusters_data;
	bool minimize_noncluster_edges;
	bool coarse_search;
	size_t cluster_overlap;
	bool use_pq;
	size_t pq_clusters;
	size_t pq_subspaces;
	ehnsw_engine_basic_fast_clusterchunks_config(
			size_t _M, size_t _M0, size_t _ef_search_mult, size_t _ef_construction,
			bool _use_cuts, size_t _min_cluster_size, size_t _max_cluster_size,
			bool _very_early_termination, bool _use_clusters_data,
			bool _minimize_noncluster_edges, bool _coarse_search,
			size_t _cluster_overlap, bool _use_pq, size_t _pq_clusters,
			size_t _pq_subspaces)
			: M(_M), M0(_M0), ef_search_mult(_ef_search_mult),
				ef_construction(_ef_construction), use_cuts(_use_cuts),
				min_cluster_size(_min_cluster_size),
				max_cluster_size(_max_cluster_size),
				very_early_termination(_very_early_termination),
				use_clusters_data(_use_clusters_data),
				minimize_noncluster_edges(_minimize_noncluster_edges),
				coarse_search(_coarse_search), cluster_overlap(_cluster_overlap),
				use_pq(_use_pq), pq_clusters(_pq_clusters),
				pq_subspaces(_pq_subspaces) {}
};

template <typename T>
struct ehnsw_engine_basic_fast_clusterchunks
		: public ann_engine<T, ehnsw_engine_basic_fast_clusterchunks<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_vertex;
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	bool use_cuts;
	size_t min_cluster_size;
	size_t max_cluster_size;
	bool very_early_termination;
	bool use_clusters_data;
	bool minimize_noncluster_edges;
	bool coarse_search;
	size_t cluster_overlap;
	bool use_pq;
	size_t pq_clusters;
	size_t pq_subspaces;
	size_t max_layer;
#ifdef RECORD_STATS
	size_t num_distcomps;
	size_t total_projected_degree;
	size_t total_clusters_checked;
	size_t total_clusters_checked_sizes;
#endif
	ehnsw_engine_basic_fast_clusterchunks(
			ehnsw_engine_basic_fast_clusterchunks_config conf)
			: rd(), gen(0), distribution(0, 1), M(conf.M), M0(conf.M0),
				ef_search_mult(conf.ef_search_mult),
				ef_construction(conf.ef_construction), use_cuts(conf.use_cuts),
				min_cluster_size(conf.min_cluster_size),
				max_cluster_size(conf.max_cluster_size),
				very_early_termination(conf.very_early_termination),
				use_clusters_data(conf.use_clusters_data),
				minimize_noncluster_edges(conf.minimize_noncluster_edges),
				coarse_search(conf.coarse_search),
				cluster_overlap(conf.cluster_overlap), use_pq(conf.use_pq),
				pq_clusters(conf.pq_clusters), pq_subspaces(conf.pq_subspaces),
				max_layer(0) {}
	using config = ehnsw_engine_basic_fast_clusterchunks_config;
	std::vector<vec<T>> all_entries;
	std::vector<std::vector<std::vector<size_t>>>
			hadj_flat; // vector -> layer -> edges
	std::vector<std::vector<size_t>>
			hadj_bottom; // vector -> edges in bottom layer
	std::vector<std::vector<std::vector<std::pair<T, size_t>>>>
			hadj_flat_with_lengths; // vector -> layer -> edges with lengths
	std::vector<vec<T>> centroids;
	std::vector<std::vector<size_t>> clusters;
	std::vector<std::vector<vec<T>>> clusters_data;
	std::vector<product_quantizer_3> clusters_searchers;
	std::vector<size_t> reverse_clusters;
	std::vector<std::vector<size_t>> hadj_bottom_projected;
	std::unique_ptr<ehnsw_engine_basic_fast<T>> coarse_searcher;
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<char> visited; // booleans
	std::vector<size_t> visited_recent;
	std::vector<std::vector<bool>> e_labels; // vertex -> cut labels (*num_cuts)
	size_t num_cuts() { return use_cuts ? e_labels[0].size() : 0; }
	std::vector<std::pair<T, size_t>>
	prune_edges(size_t layer, size_t from, std::vector<std::pair<T, size_t>> to);
	template <bool use_bottomlayer, bool use_clusters>
	std::vector<std::pair<T, size_t>>
	query_k_at_layer(const vec<T>& q, size_t layer,
									 const std::vector<size_t>& entry_points, size_t k);
	std::vector<std::pair<T, size_t>>
	query_k_at_bottom_via_clusters(const vec<T>& q, size_t layer,
																 const std::vector<size_t>& entry_points,
																 size_t k);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() {
		return use_cuts ? "EHNSW Engine Basic Fast Cluster-Chunks"
										: "HNSW Engine Basic Fast Cluster-Chunks";
	}
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, M0);
		add_param(pl, ef_search_mult);
		add_param(pl, ef_construction);
		add_param(pl, use_cuts);
		add_param(pl, min_cluster_size);
		add_param(pl, max_cluster_size);
		add_param(pl, very_early_termination);
		add_param(pl, use_clusters_data);
		add_param(pl, minimize_noncluster_edges);
		add_param(pl, coarse_search);
		add_param(pl, cluster_overlap);
		add_param(pl, use_pq);
		add_param(pl, pq_clusters);
		add_param(pl, pq_subspaces);
#ifdef RECORD_STATS
		add_param(pl, num_distcomps);
		add_param(pl, total_projected_degree);
		add_param(pl, total_clusters_checked);
		add_param(pl, total_clusters_checked_sizes);
#endif
		return pl;
	}
	bool generate_elabel() {
		std::uniform_int_distribution<> int_distribution(0, 1);
		return int_distribution(gen);
	}
};

template <typename T>
std::vector<std::pair<T, size_t>>
ehnsw_engine_basic_fast_clusterchunks<T>::prune_edges(
		size_t layer, size_t from, std::vector<std::pair<T, size_t>> to) {
	auto edge_count_mult = M;
	if (layer == 0)
		edge_count_mult = M0;

	// reference impl vs paper difference
	if (to.size() <= edge_count_mult) {
		return to;
	}

	sort(to.begin(), to.end());
	std::vector<std::pair<T, size_t>> ret;
	std::vector<bool> bins(layer > 0 ? 0 : edge_count_mult - num_cuts());
	for (const auto& md : to) {
		_mm_prefetch(&all_entries[md.second], _MM_HINT_T0);
		if (ret.size() >= edge_count_mult)
			break;
		bool choose = true;
		for (const auto& md_chosen : ret) {
			if (md.first == md_chosen.first ||
					dist2(all_entries[md.second], all_entries[md_chosen.second]) <=
							md.first) {
				choose = false;
				break;
			}
		}
		if (layer == 0 && ret.size() + num_cuts() >= edge_count_mult) {
			bool found_bin = false;
			for (size_t bin = 0; bin < bins.size(); ++bin) {
				if (!bins[bin] && e_labels[md.second][bin] != e_labels[from][bin]) {
					bins[bin] = true;
					found_bin = true;
				}
			}
			if (!found_bin) {
				choose = false;
			}
		}
		if (choose) {
			ret.emplace_back(md);
		}
	}

	return ret;
}

template <typename T>
void ehnsw_engine_basic_fast_clusterchunks<T>::_store_vector(const vec<T>& v) {
	size_t v_index = all_entries.size();
	all_entries.push_back(v);

	e_labels.emplace_back();
	for (size_t cut = 0; cut < M0 - 2 * 10; ++cut)
		e_labels.back().emplace_back(generate_elabel());

	size_t new_max_layer = floor(-log(distribution(gen)) * 1 / log(double(M)));
	// size_t new_max_layer = 0;

	hadj_flat_with_lengths.emplace_back();
	for (size_t layer = 0; layer <= new_max_layer; ++layer) {
		hadj_flat_with_lengths[v_index].emplace_back();
	}

	auto convert_el = [](std::vector<std::pair<T, size_t>> el) constexpr {
		std::vector<size_t> ret;
		ret.reserve(el.size());
		for (auto& [_, val] : el) {
			ret.emplace_back(val);
		}
		return ret;
	};

	// get kNN for each layer
	std::vector<std::vector<std::pair<T, size_t>>> kNN_per_layer;
	if (all_entries.size() > 1) {
		std::vector<size_t> cur = {starting_vertex};
		{
			size_t entry_point = starting_vertex;
			T ep_dist = dist2(v, all_entries[entry_point]);
			for (size_t layer = max_layer - 1; layer > new_max_layer; --layer) {
				bool changed = true;
				while (changed) {
					changed = false;
					for (auto& neighbour : hadj_flat[entry_point][layer]) {
						_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
						T neighbour_dist = dist2(v, all_entries[neighbour]);
						if (neighbour_dist < ep_dist) {
							entry_point = neighbour;
							ep_dist = neighbour_dist;
							changed = true;
						}
					}
				}
			}
			cur = {entry_point};
		}
		for (int layer = std::min(new_max_layer, max_layer - 1); layer >= 0;
				 --layer) {
			kNN_per_layer.emplace_back(
					query_k_at_layer<false, false>(v, layer, cur, ef_construction));
			cur.clear();
			for (auto& md : kNN_per_layer.back()) {
				cur.emplace_back(md.second);
			}
			cur.resize(1); // present in reference impl, but not in hnsw paper
		}

		std::reverse(kNN_per_layer.begin(), kNN_per_layer.end());
	}

	// add the found edges to the graph
	for (size_t layer = 0; layer < std::min(max_layer, new_max_layer + 1);
			 ++layer) {
		hadj_flat_with_lengths[v_index][layer] =
				prune_edges(layer, v_index, kNN_per_layer[layer]);
		//  add bidirectional connections, prune if necessary
		for (auto& md : kNN_per_layer[layer]) {
			bool edge_exists = false;
			for (auto& md_other : hadj_flat_with_lengths[md.second][layer]) {
				if (md_other.second == v_index) {
					edge_exists = true;
				}
			}
			if (!edge_exists) {
				hadj_flat_with_lengths[md.second][layer].emplace_back(md.first,
																															v_index);
				hadj_flat_with_lengths[md.second][layer] = prune_edges(
						layer, md.second, hadj_flat_with_lengths[md.second][layer]);
				hadj_flat[md.second][layer] =
						convert_el(hadj_flat_with_lengths[md.second][layer]);
				if (layer == 0)
					hadj_bottom[md.second] = hadj_flat[md.second][layer];
			}
		}
	}

	// add new layers if necessary
	while (new_max_layer >= max_layer) {
		++max_layer;
		starting_vertex = v_index;
	}

	visited.emplace_back();
	hadj_flat.emplace_back();
	hadj_bottom.emplace_back();
	hadj_bottom[v_index] = convert_el(hadj_flat_with_lengths[v_index][0]);
	for (size_t layer = 0; layer <= new_max_layer; ++layer) {
		hadj_flat[v_index].emplace_back();
		hadj_flat[v_index][layer] =
				convert_el(hadj_flat_with_lengths[v_index][layer]);
	}
}

template <typename T> void ehnsw_engine_basic_fast_clusterchunks<T>::_build() {
	assert(all_entries.size() > 0);
#ifdef RECORD_STATS
	total_projected_degree = 0;
	total_clusters_checked = 0;
	total_clusters_checked_sizes = 0;
#endif

	// start by setting centroids to the values of the second-to-last layer
	for (size_t i = 0; i < all_entries.size(); ++i) {
		if (hadj_flat[i].size() > 1) {
			centroids.emplace_back(all_entries[i]);
		}
	}
	auto assign_to_clusters = [&]() {
		ehnsw_engine_basic_fast<T> sub_engine(ehnsw_engine_basic_fast_config(
				M, M0, ef_search_mult, ef_construction, use_cuts));
		for (auto& v : centroids)
			sub_engine.store_vector(v);
		sub_engine.build();
		clusters.clear();
		clusters.resize(centroids.size());
		for (size_t i = 0; i < all_entries.size(); ++i) {
			auto sub_engine_results =
					sub_engine.query_k(all_entries[i], cluster_overlap);
			for (size_t j = 0;
					 j < std::min(cluster_overlap, sub_engine_results.size()); ++j) {
				clusters[sub_engine_results[j]].emplace_back(i);
			}
		}
	};
	auto compute_centroids = [&]() {
		centroids.resize(clusters.size());
		for (size_t centroid_index = 0; centroid_index < centroids.size();
				 ++centroid_index) {
			auto& centroid = centroids[centroid_index];
			centroid.clear();
			for (size_t elem_index : clusters[centroid_index]) {
				centroid += all_entries[elem_index];
			}
			if (!clusters[centroid_index].empty()) {
				centroid /= clusters[centroid_index].size();
			}
		}
	};
	const size_t max_iters = M / 2;
	vec_generator<T> rvgen(all_entries[0].size());
	for (size_t iter = 0; iter < max_iters; ++iter) {
		assign_to_clusters();
		// loosely enforce min/max cluster sizes
		std::vector<std::vector<size_t>> clusters_new;
		for (size_t cluster_index = 0; cluster_index < clusters.size();
				 ++cluster_index) {
			// enforce min cluster size by deleting small clusters
			if (clusters[cluster_index].size() >= min_cluster_size) {
				// enforce max cluster size by splitting big clusters
				if (clusters[cluster_index].size() <= max_cluster_size) {
					clusters_new.emplace_back(clusters[cluster_index]);
				} else {
					// partition with a random hyperplane
					auto project_vec = rvgen.random_vec(); // normal vector of hyperplane
					clusters_new.emplace_back();
					clusters_new.emplace_back();
					for (size_t entry_index : clusters[cluster_index]) {
						clusters_new[clusters_new.size() - 1 -
												 ((all_entries[entry_index] - centroids[cluster_index])
															.dot(project_vec) > 0)]
								.emplace_back(entry_index);
					}
				}
			}
		}
		if (!clusters_new.empty()) // don't let all clusters be deleted
			clusters = clusters_new;
		compute_centroids();
	}
	assign_to_clusters();
	if (cluster_overlap == 1) {
		reverse_clusters.resize(all_entries.size());
		for (size_t i = 0; i < clusters.size(); ++i) {
			for (size_t j : clusters[i])
				reverse_clusters[j] = i;
		}
		if (minimize_noncluster_edges) {
			std::cout << "About to start minimizing noncluster edges" << std::endl;
			// std::vector<size_t> next_reverse_clusters(all_entries.size());
			for (size_t i = 0; i < max_iters; ++i) {
				std::cout << "Iteration no. " << i << " of minimizing noncluster edges"
									<< std::endl;
				// sort clusters by increasing size
				std::vector<std::pair<size_t, size_t>> cluster_order;
				for (size_t i = 0; i < clusters.size(); ++i) {
					cluster_order.emplace_back(clusters[i].size(), i);
				}
				std::sort(cluster_order.begin(), cluster_order.end());
				for (auto [_, cluster_index] : cluster_order) {
					size_t num_removed = 0;
					for (size_t data_index : clusters[cluster_index]) {
						// decide if data_index should be moved to a different cluster
						robin_hood::unordered_flat_map<size_t, double> cluster_connectivity;
						for (const auto& [adjacent_d2, adjacent_data_index] :
								 hadj_flat_with_lengths[data_index][0]) {
							size_t adjacent_cluster_index =
									reverse_clusters[adjacent_data_index];
							// TODO consider removing this restriction or replacing it with a
							// looser one
							if (clusters[adjacent_cluster_index].size() <
									clusters[cluster_index].size() - num_removed)
								cluster_connectivity[adjacent_cluster_index] +=
										1.0 / adjacent_d2;
						}
						double best_modularity_gain = 0.0;
						size_t best_cluster = cluster_index;
						for (const auto& [adjacent_cluster, connectivity] :
								 cluster_connectivity) {
							if (adjacent_cluster != cluster_index) {
								double expected_connectivity =
										clusters[adjacent_cluster].size() /
										dist2(all_entries[data_index], centroids[adjacent_cluster]);

								double modularity_gain = connectivity - expected_connectivity;
								if (modularity_gain > best_modularity_gain) {
									best_modularity_gain = modularity_gain;
									best_cluster = adjacent_cluster;
								}
							}
						}
						reverse_clusters[data_index] = best_cluster;
						if (cluster_index != reverse_clusters[data_index]) {
							clusters[reverse_clusters[data_index]].emplace_back(data_index);
							++num_removed;
						}
					}
					std::vector<size_t> new_cluster_data;
					for (size_t data_index : clusters[cluster_index])
						if (reverse_clusters[data_index] == cluster_index)
							new_cluster_data.emplace_back(data_index);
					clusters[cluster_index] = new_cluster_data;
				}
				clusters.clear();
				clusters.resize(centroids.size());
				for (size_t data_index = 0; data_index < all_entries.size();
						 ++data_index) {
					clusters[reverse_clusters[data_index]].emplace_back(data_index);
				}
				compute_centroids();
			}
		}
	}
	if (use_clusters_data) {
		for (size_t cluster_index = 0; cluster_index < clusters.size();
				 ++cluster_index) {
			clusters_data.emplace_back();
			for (size_t data_index : clusters[cluster_index]) {
				clusters_data[cluster_index].emplace_back(all_entries[data_index]);
			}
		}
	}

	if (coarse_search) {
		coarse_searcher = std::make_unique<ehnsw_engine_basic_fast<T>>(
				ehnsw_engine_basic_fast_config(M, M0, ef_search_mult, ef_construction,
																			 use_cuts));
		for (auto& v : centroids)
			coarse_searcher->store_vector(v);
		coarse_searcher->build();

		if (use_pq) {
			std::cout << "About to start pq indexing" << std::endl;
			for (size_t cluster_index = 0; cluster_index < clusters.size();
					 ++cluster_index) {
				std::vector<typename vec<T>::Underlying> residuals;
				for (size_t i : clusters[cluster_index]) {
					residuals.emplace_back(
							(all_entries[i] - centroids[cluster_index]).get_underlying());
				}
				clusters_searchers.emplace_back(residuals, pq_clusters,
																				all_entries[0].size() / pq_subspaces);
			}
			std::cout << "Just finished pq indexing" << std::endl;
		}
	} else {
		hadj_bottom_projected.resize(all_entries.size());
		for (size_t i = 0; i < all_entries.size(); ++i) {
			robin_hood::unordered_flat_set<size_t> added = {reverse_clusters[i]};
			for (size_t j : hadj_bottom[i]) {
				size_t cluster_index = reverse_clusters[j];
				if (!added.contains(cluster_index)) {
					hadj_bottom_projected[i].emplace_back(cluster_index);
					added.insert(cluster_index);
				}
			}
#ifdef RECORD_STATS
			total_projected_degree += hadj_bottom_projected[i].size();
#endif
		}
	}

#ifdef RECORD_STATS
	// reset before queries
	num_distcomps = 0;
#endif

	std::cout << "Finished build" << std::endl;
}

template <typename T>
template <bool use_bottomlayer, bool use_clusters>
std::vector<std::pair<T, size_t>>
ehnsw_engine_basic_fast_clusterchunks<T>::query_k_at_layer(
		const vec<T>& q, size_t layer, const std::vector<size_t>& entry_points,
		size_t k) {
	using measured_data = std::pair<T, size_t>;

	auto get_vertex = [&](const size_t& index) constexpr->std::vector<size_t>& {
		if constexpr (use_bottomlayer) {
			return hadj_bottom[index];
		} else {
			return hadj_flat[index][layer];
		}
	};

	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	for (auto& entry_point : entry_points) {
#ifdef RECORD_STATS
		++num_distcomps;
#endif
		entry_points_with_dist.emplace_back(dist2(q, all_entries[entry_point]),
																				entry_point);
	}

	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(best_elem)>
			candidates(entry_points_with_dist.begin(), entry_points_with_dist.end(),
								 best_elem);
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest(entry_points_with_dist.begin(), entry_points_with_dist.end(),
							worst_elem);
	while (nearest.size() > k)
		nearest.pop();

	for (auto& entry_point : entry_points) {
		visited[entry_point] = true;
		visited_recent.emplace_back(entry_point);
	}

	while (!candidates.empty()) {
		auto cur = candidates.top();
		candidates.pop();
		if (cur.first > nearest.top().first && nearest.size() == k) {
			break;
		}
		std::vector<size_t> neighbour_list;
		if constexpr (use_clusters && use_bottomlayer) {
			for (size_t cluster_neighbour : hadj_bottom_projected[cur.second]) {
				for (size_t neighbour : clusters[cluster_neighbour]) {
					if (!visited[neighbour]) {
						neighbour_list.emplace_back(neighbour);
						visited[neighbour] = true;
						visited_recent.emplace_back(neighbour);
					}
				}
			}
		} else
			for (size_t neighbour : get_vertex(cur.second))
				if (!visited[neighbour]) {
					neighbour_list.emplace_back(neighbour);
					visited[neighbour] = true;
					visited_recent.emplace_back(neighbour);
				}
		constexpr size_t in_advance = 4;
		constexpr size_t in_advance_extra = 2;
		auto do_loop_prefetch = [&](size_t i) constexpr {
#ifdef DIM
			for (size_t mult = 0; mult < DIM * sizeof(T) / 64; ++mult)
				_mm_prefetch(((char*)&all_entries[neighbour_list[i]]) + mult * 64,
										 _MM_HINT_T0);
#endif
		};
		for (size_t next_i_pre = 0;
				 next_i_pre < std::min(in_advance, neighbour_list.size());
				 ++next_i_pre) {
			do_loop_prefetch(next_i_pre);
		}
		auto loop_iter = [&]<bool inAdvanceIter, bool inAdvanceIterExtra>(
				size_t next_i) constexpr {
			if constexpr (inAdvanceIterExtra) {
				_mm_prefetch(&neighbour_list[next_i + in_advance + in_advance_extra],
										 _MM_HINT_T0);
			}
			if constexpr (inAdvanceIter) {
				do_loop_prefetch(next_i + in_advance);
			}
			const auto& next = neighbour_list[next_i];
			// if (!visited[next]) {
			// visited[next] = true;
			// visited_recent.emplace_back(next);
#ifdef RECORD_STATS
			++num_distcomps;
#endif
			T d_next = dist2(q, all_entries[next]);
			if (nearest.size() < k || d_next < nearest.top().first) {
				candidates.emplace(d_next, next);
				nearest.emplace(d_next, next);
				if (nearest.size() > k)
					nearest.pop();
			}
			//}
		};
		size_t next_i = 0;
		for (; next_i + in_advance + in_advance_extra < neighbour_list.size();
				 ++next_i) {
			loop_iter.template operator()<true, true>(next_i);
		}
		for (; next_i + in_advance < neighbour_list.size(); ++next_i) {
			loop_iter.template operator()<true, false>(next_i);
		}
		for (; next_i < neighbour_list.size(); ++next_i) {
			loop_iter.template operator()<false, false>(next_i);
		}
	}
	for (auto& v : visited_recent)
		visited[v] = false;
	visited_recent.clear();
	std::vector<measured_data> ret;
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top());
		nearest.pop();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<std::pair<T, size_t>>
ehnsw_engine_basic_fast_clusterchunks<T>::query_k_at_bottom_via_clusters(
		const vec<T>& q, size_t layer,
		const std::vector<size_t>& initial_entry_points, size_t k) {
	using measured_data = std::pair<T, size_t>;

	auto worst_elem = [](const measured_data& a, const measured_data& b) {
		return a.first < b.first;
	};
	auto best_elem = [](const measured_data& a, const measured_data& b) {
		return a.first > b.first;
	};
	std::vector<measured_data> entry_points_with_dist;
	for (auto& entry_point : initial_entry_points) {
#ifdef RECORD_STATS
		++num_distcomps;
#endif
		size_t cluster_index = reverse_clusters[entry_point];
		visited[cluster_index] = true;
		visited_recent.emplace_back(cluster_index);
		for (size_t entry_index : clusters[cluster_index]) {
			T d_entry_index = dist2(q, all_entries[entry_index]);
			entry_points_with_dist.emplace_back(d_entry_index, entry_index);
		}
	}

	//, std::vector<measured_data>, decltype(best_elem)>
	//, best_elem);
	std::set<measured_data> candidates(entry_points_with_dist.begin(),
																		 entry_points_with_dist.end());
	std::priority_queue<measured_data, std::vector<measured_data>,
											decltype(worst_elem)>
			nearest(entry_points_with_dist.begin(), entry_points_with_dist.end(),
							worst_elem);
	while (nearest.size() > k)
		nearest.pop();

	while (!candidates.empty()) {
		auto cur = candidates.extract(candidates.begin()).value();
		// candidates.pop_front();
		if (cur.first > nearest.top().first && nearest.size() == k) {
			break;
		}
		auto loop_with_prefetches =
				[&](const std::vector<size_t>& member_list,
						const std::vector<vec<T>>& vec_list,
						std::function<void(const size_t&)> inner_loop_fn) {
					constexpr size_t in_advance = 4;
					constexpr size_t in_advance_extra = 2;
					auto do_loop_prefetch = [&](size_t i) constexpr {
#ifdef DIM
						for (size_t mult = 0; mult < DIM * sizeof(T) / 64; ++mult)
							_mm_prefetch(((char*)&vec_list[member_list[i]]) + mult * 64,
													 _MM_HINT_T0);
#endif
					};
					for (size_t next_i_pre = 0;
							 next_i_pre < std::min(in_advance, member_list.size());
							 ++next_i_pre) {
						do_loop_prefetch(next_i_pre);
					}
					auto loop_iter = [&]<bool inAdvanceIter, bool inAdvanceIterExtra>(
							size_t next_i) constexpr {
						if constexpr (inAdvanceIterExtra) {
							_mm_prefetch(&member_list[next_i + in_advance + in_advance_extra],
													 _MM_HINT_T0);
						}
						if constexpr (inAdvanceIter) {
							do_loop_prefetch(next_i + in_advance);
						}
						const auto& next = member_list[next_i];
						inner_loop_fn(next);
					};
					size_t next_i = 0;
					for (; next_i + in_advance + in_advance_extra < member_list.size();
							 ++next_i) {
						loop_iter.template operator()<true, true>(next_i);
					}
					for (; next_i + in_advance < member_list.size(); ++next_i) {
						loop_iter.template operator()<true, false>(next_i);
					}
					for (; next_i < member_list.size(); ++next_i) {
						loop_iter.template operator()<false, false>(next_i);
					}
				};
		bool found_continuation = nearest.size() < k;
		for (size_t cluster_index : hadj_bottom_projected[cur.second]) {
			if (!visited[cluster_index]) {
				visited[cluster_index] = true;
				visited_recent.emplace_back(cluster_index);
#ifdef RECORD_STATS
				++total_clusters_checked;
				total_clusters_checked_sizes += clusters[cluster_index].size();
#endif
				std::vector<measured_data> to_check;
				to_check.reserve(clusters[cluster_index].size());
				if (use_clusters_data) {
					for (size_t inside_cluster_index = 0;
							 inside_cluster_index < clusters[cluster_index].size();
							 ++inside_cluster_index) {
						size_t next = clusters[cluster_index][inside_cluster_index];
#ifdef RECORD_STATS
						++num_distcomps;
#endif
						T d_next =
								dist2(q, clusters_data[cluster_index][inside_cluster_index]);
						to_check.emplace_back(d_next, next);
					}
				} else {
					loop_with_prefetches(clusters[cluster_index], all_entries,
															 [&](const size_t& next) {
#ifdef RECORD_STATS
																 ++num_distcomps;
#endif
																 T d_next = dist2(q, all_entries[next]);
																 to_check.emplace_back(d_next, next);
															 });
				}
				for (const auto& [d_next, next] : to_check) {
					if (nearest.size() < k || d_next < nearest.top().first) {
						nearest.emplace(d_next, next);
						if (nearest.size() > k)
							nearest.pop();
						found_continuation = true;
						candidates.emplace(d_next, next);
					}
				}
			}
			while (candidates.size() > k)
				candidates.erase(std::prev(candidates.end()));
		}
		if (very_early_termination && !found_continuation) {
			break;
		}
	}
	for (auto& v : visited_recent)
		visited[v] = false;
	visited_recent.clear();
	std::vector<measured_data> ret;
	while (!nearest.empty()) {
		ret.emplace_back(nearest.top());
		nearest.pop();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t>
ehnsw_engine_basic_fast_clusterchunks<T>::_query_k(const vec<T>& q, size_t k) {
	if (coarse_search) {
		std::vector<size_t> clusters_to_check =
				coarse_searcher->query_k(q, k * ef_search_mult);
		using measured_data = std::pair<T, size_t>;
		std::vector<measured_data> results;
		if (use_pq) {
			for (size_t cluster_index : clusters_to_check) {
				std::vector<T> distances =
						clusters_searchers[cluster_index].compute_distances(
								q - centroids[cluster_index]);
				for (size_t inside_cluster_index = 0;
						 inside_cluster_index < clusters[cluster_index].size();
						 ++inside_cluster_index) {
					results.emplace_back(distances[inside_cluster_index],
															 clusters[cluster_index][inside_cluster_index]);
				}
			}
		} else if (use_clusters_data) {
			for (size_t cluster_index : clusters_to_check)
				for (size_t inside_cluster_index = 0;
						 inside_cluster_index < clusters[cluster_index].size();
						 ++inside_cluster_index) {
					size_t next = clusters[cluster_index][inside_cluster_index];
#ifdef RECORD_STATS
					++num_distcomps;
#endif
					T d_next =
							dist2(q, clusters_data[cluster_index][inside_cluster_index]);
					results.emplace_back(d_next, next);
				}
		} else {
			for (size_t cluster_index : clusters_to_check)
				for (size_t inside_cluster_index = 0;
						 inside_cluster_index < clusters[cluster_index].size();
						 ++inside_cluster_index) {
					size_t next = clusters[cluster_index][inside_cluster_index];
#ifdef RECORD_STATS
					++num_distcomps;
#endif
					T d_next = dist2(q, all_entries[next]);
					results.emplace_back(d_next, next);
				}
		}
		std::sort(results.begin(), results.end(),
							[](const measured_data& a, const measured_data& b) {
								return a.second < b.second;
							});
		auto last = std::unique(results.begin(), results.end(),
														[](const measured_data& a, const measured_data& b) {
															return a.second == b.second;
														});
		results.erase(last, results.end());
		std::partial_sort(results.begin(),
											results.begin() + std::min(results.size(), k),
											results.end());

		size_t ret_size = std::min(k, results.size());
		results.resize(ret_size);
		std::vector<size_t> ret;
		for (size_t i = 0; i < results.size() && i < k; ++i) {
			ret.emplace_back(results[i].second);
		}
		return ret;
	}

	size_t entry_point = starting_vertex;
#ifdef RECORD_STATS
	++num_distcomps;
#endif
	T ep_dist = dist2(all_entries[entry_point], q);
	for (size_t layer = max_layer - 1; layer > 0; --layer) {
		bool changed = true;
		while (changed) {
			changed = false;
			for (auto& neighbour : hadj_flat[entry_point][layer]) {
				_mm_prefetch(&all_entries[neighbour], _MM_HINT_T0);
#ifdef RECORD_STATS
				++num_distcomps;
#endif
				T neighbour_dist = dist2(q, all_entries[neighbour]);
				if (neighbour_dist < ep_dist) {
					entry_point = neighbour;
					ep_dist = neighbour_dist;
					changed = true;
				}
			}
		}
	}

	// auto ret_combined =
	// 		query_k_at_layer<true, true>(q, 0, {entry_point}, k * ef_search_mult);
	auto ret_combined =
			query_k_at_bottom_via_clusters(q, 0, {entry_point}, k * ef_search_mult);
	if (ret_combined.size() > k)
		ret_combined.resize(k);
	std::vector<size_t> ret;
	for (size_t i = 0; i < ret_combined.size() && i < k; ++i) {
		ret.emplace_back(ret_combined[i].second);
	}
	return ret;
}
