#pragma once

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ann_engine.h"
#include "robin_hood.h"
#include "topk_t.h"

struct ehnsw_engine_basic_clustern_config {
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	size_t num_clusters;
	size_t cluster_size;
	ehnsw_engine_basic_clustern_config(size_t _M, size_t _M0,
																		 size_t _ef_search_mult,
																		 size_t _ef_construction,
																		 size_t _num_clusters, size_t _cluster_size)
			: M(_M), M0(_M0), ef_search_mult(_ef_search_mult),
				ef_construction(_ef_construction), num_clusters(_num_clusters),
				cluster_size(_cluster_size) {}
};

template <typename T>
struct ehnsw_engine_basic_clustern
		: public ann_engine<T, ehnsw_engine_basic_clustern<T>> {
	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> distribution;
	size_t starting_vertex;
	size_t M;
	size_t M0;
	size_t ef_search_mult;
	size_t ef_construction;
	size_t num_clusters;
	size_t cluster_size;
	size_t max_layer;
#ifdef RECORD_STATS
	size_t num_distcomps;
#endif
	ehnsw_engine_basic_clustern(ehnsw_engine_basic_clustern_config conf)
			: rd(), gen(0), distribution(0, 1), M(conf.M), M0(conf.M0),
				ef_search_mult(conf.ef_search_mult),
				ef_construction(conf.ef_construction), num_clusters(conf.num_clusters),
				cluster_size(conf.cluster_size), max_layer(0) {}
	std::vector<vec<T>> all_entries;
	std::vector<std::vector<std::vector<size_t>>>
			hadj_flat; // vector -> layer -> edges
	std::vector<std::vector<size_t>>
			hadj_bottom; // vector -> edges in bottom layer
	std::vector<std::vector<size_t>>
			cluster_centres; // vector -> bottom-layer edges to cluster centres
	std::vector<std::vector<std::vector<size_t>>>
			clusters; // vector -> cluster index -> bottom-layer edges in cluster
	std::vector<std::vector<std::vector<std::pair<T, size_t>>>>
			hadj_flat_with_lengths; // vector -> layer -> edges with lengths
	void _store_vector(const vec<T>& v);
	void _build();
	std::vector<char> visited; // booleans
	std::vector<size_t> visited_recent;
	std::vector<std::vector<bool>> e_labels; // vertex -> cut labels (*num_cuts)
	size_t num_cuts() { return e_labels[0].size(); }
	std::vector<std::pair<T, size_t>>
	prune_edges(size_t layer, size_t from, std::vector<std::pair<T, size_t>> to);
	template <bool use_bottomlayer>
	std::vector<std::pair<T, size_t>>
	query_k_at_layer(const vec<T>& q, size_t layer,
									 const std::vector<size_t>& entry_points, size_t k);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	std::vector<std::pair<T, size_t>> query_k_combined(const vec<T>& v, size_t k);
	const std::string _name() { return "(noE)HNSW Engine Basic Cluster-N"; }
	const param_list_t _param_list() {
		param_list_t pl;
		add_param(pl, M);
		add_param(pl, M0);
		add_param(pl, ef_search_mult);
		add_param(pl, ef_construction);
		add_param(pl, num_clusters);
		add_param(pl, cluster_size);
#ifdef RECORD_STATS
		add_param(pl, num_distcomps);
#endif
		return pl;
	}
	bool generate_elabel() {
		std::uniform_int_distribution<> int_distribution(0, 1);
		return int_distribution(gen);
	}
};

template <typename T>
std::vector<std::pair<T, size_t>> ehnsw_engine_basic_clustern<T>::prune_edges(
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
				if (!bins[bin]) {
					//&& e_labels[md.second][bin] != e_labels[from][bin]) {
					bins[bin] = true;
					found_bin = true;
				}
			}
			if (!found_bin) {
				choose = false;
			}
		}
		// if (ret.size() + num_cuts() >= edge_count_mult) {
		//	if (e_labels[md.second][ret.size() + num_cuts() - edge_count_mult] ==
		//			e_labels[from][ret.size() + num_cuts() - edge_count_mult])
		//		choose = false;
		// }
		if (choose) {
			ret.emplace_back(md);
		}
	}

	return ret;
}

template <typename T>
void ehnsw_engine_basic_clustern<T>::_store_vector(const vec<T>& v) {
	size_t v_index = all_entries.size();
	all_entries.push_back(v);

	if (v_index % 5000 == 0) {
		std::cerr << "Inserting entry " << v_index << std::endl;
	}

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
					query_k_at_layer<false>(v, layer, cur, ef_construction));
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

template <typename T> void ehnsw_engine_basic_clustern<T>::_build() {
	assert(all_entries.size() > 0);

	// cluster edges
	cluster_centres.resize(all_entries.size());
	clusters.resize(all_entries.size());
	for (size_t v = 0; v < all_entries.size(); ++v) {
		if (v % 10000 == 0) {
			std::cerr << double(v) / all_entries.size() * 100
								<< "\% of the way through clustering" << std::endl;
		}
		// choose cluster centres
		std::vector<size_t> outgoing_edges = hadj_bottom[v];
		std::shuffle(outgoing_edges.begin(), outgoing_edges.end(), gen);
		// first num_clusters edges become cluster centres
		cluster_centres[v] = outgoing_edges;
		if (cluster_centres[v].size() > num_clusters)
			cluster_centres[v].resize(num_clusters);
		clusters[v].resize(cluster_centres[v].size());
		if (cluster_centres[v].size() == outgoing_edges.size())
			continue; // nothing left to put in clusters

		// only the rest of them get clustered
		std::vector<size_t> remaining_outgoing_edges;
		for (size_t i = cluster_centres[v].size(); i < outgoing_edges.size(); ++i)
			remaining_outgoing_edges.emplace_back(outgoing_edges[i]);
		// for each remaining vector, rank the cluster centres
		std::vector<std::vector<std::pair<T, size_t>>> best_clusters(
				remaining_outgoing_edges.size());
		for (size_t i = 0; i < remaining_outgoing_edges.size(); ++i) {
			for (size_t j = 0; j < cluster_centres[v].size(); ++j) {
				best_clusters[i].emplace_back(
						dist2(all_entries[remaining_outgoing_edges[i]],
									all_entries[cluster_centres[v][j]]),
						j);
			}
			std::sort(best_clusters[i].begin(), best_clusters[i].end());
		}
		// greedily assign to clusters until they are all full
		// with good params this should guarantee that everything gets assigned
		// to some cluster
		bool newly_assigned = true;
		std::vector<size_t> next_cluster_to_try(remaining_outgoing_edges.size());
		while (newly_assigned) {
			newly_assigned = false;
			for (size_t i = 0; i < remaining_outgoing_edges.size(); ++i) {
				for (; next_cluster_to_try[i] < best_clusters[i].size();
						 ++next_cluster_to_try[i]) {
					size_t ci_to_try = best_clusters[i][next_cluster_to_try[i]].second;
					if (clusters[v][ci_to_try].size() < cluster_size) {
						newly_assigned = true;
						clusters[v][ci_to_try].emplace_back(remaining_outgoing_edges[i]);
						++next_cluster_to_try[i];
						break;
					}
				}
			}
		}
	}

#ifdef RECORD_STATS
	// reset before queries
	num_distcomps = 0;
#endif
}

template <typename T>
template <bool use_bottomlayer>
std::vector<std::pair<T, size_t>>
ehnsw_engine_basic_clustern<T>::query_k_at_layer(
		const vec<T>& q, size_t layer, const std::vector<size_t>& entry_points,
		size_t k) {
	using measured_data = std::pair<T, size_t>;

	auto get_vertex = [&](const size_t& index) constexpr->std::vector<size_t>& {
		if constexpr (use_bottomlayer) {
			return cluster_centres[index];
			// return hadj_bottom[index];
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
		constexpr size_t in_advance = 4;
		constexpr size_t in_advance_extra = 2;
		auto process_neighbour_list = [&](
				const std::vector<size_t>& neighbour_list) constexpr {
			std::vector<std::pair<T, size_t>> to_consider;
			auto do_loop_prefetch = [&](size_t i) constexpr {
#ifdef DIM
				for (size_t mult = 0; mult < DIM * sizeof(T) / 64; ++mult)
					_mm_prefetch(((char*)&all_entries[neighbour_list[i]]) + mult * 64,
											 _MM_HINT_T0);
#endif
				_mm_prefetch(&visited[neighbour_list[i]], _MM_HINT_T0);
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
				if (!visited[next]) {
					visited[next] = true;
					visited_recent.emplace_back(next);
#ifdef RECORD_STATS
					++num_distcomps;
#endif
					T d_next = dist2(q, all_entries[next]);
					to_consider.emplace_back(d_next, next);
				}
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
			return to_consider;
		};
		auto consider_list = [&](const std::vector<std::pair<T, size_t>>& cl) {
			for (auto& [d_next, next] : cl) {
				if (nearest.size() < k || d_next < nearest.top().first) {
					candidates.emplace(d_next, next);
					nearest.emplace(d_next, next);
					if (nearest.size() > k)
						nearest.pop();
				}
			}
		};
		auto candidate_list = process_neighbour_list(get_vertex(cur.second));
		consider_list(candidate_list);
		if constexpr (use_bottomlayer) {
			auto cluster_index = std::distance(
					candidate_list.begin(),
					std::min_element(candidate_list.begin(), candidate_list.end()));
			auto sub_candidate_list =
					process_neighbour_list(clusters[cur.second][cluster_index]);
			consider_list(sub_candidate_list);
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
std::vector<size_t> ehnsw_engine_basic_clustern<T>::_query_k(const vec<T>& q,
																														 size_t k) {
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

	auto ret_combined =
			query_k_at_layer<true>(q, 0, {entry_point}, k * ef_search_mult);
	if (ret_combined.size() > k)
		ret_combined.resize(k);
	std::vector<size_t> ret;
	for (size_t i = 0; i < ret_combined.size() && i < k; ++i) {
		ret.emplace_back(ret_combined[i].second);
	}
	return ret;
}

template <typename T>
std::vector<std::pair<T, size_t>>
ehnsw_engine_basic_clustern<T>::query_k_combined(const vec<T>& q, size_t k) {
	size_t entry_point = starting_vertex;
#ifdef RECORD_STATS
	++num_distcomps;
#endif
	T ep_dist = dist2(all_entries[entry_point], q);
	for (int layer = max_layer - 1; layer >= 0; --layer) {
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

	auto ret_combined =
			query_k_at_layer<true>(q, 0, {entry_point}, k * ef_search_mult);
	if (ret_combined.size() > k)
		ret_combined.resize(k);
	return ret_combined;
}