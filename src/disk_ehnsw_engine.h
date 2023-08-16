#pragma once

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "ann_engine.h"
#include "ehnsw_engine_2.h"
#include "memory_mapped_vector.h"
#include "robin_hood.h"

struct disk_ehnsw_engine_config {
	ehnsw_engine_2_config subconf;
	size_t max_mb_all_accessor;
	size_t max_mb_index_accessor;
	std::string filename;
	disk_ehnsw_engine_config(ehnsw_engine_2_config _subconf,
													 size_t _max_mb_all_accessor,
													 size_t _max_mb_index_accessor,
													 const std::string& _filename)
			: subconf(_subconf), max_mb_all_accessor(_max_mb_all_accessor),
				max_mb_index_accessor(_max_mb_index_accessor), filename(_filename) {}
};

// Disk index contains the following (in order):
// size_t num_vecs, size_t dim, size_t num_nodes, size_t num_edges
// List<Node> nodes
// -- compacted list of nodes at all levels (in format described below)
// List<size_t> edges
// -- compacted list of all outgoing edges
//    (each node's outgoing edges are contiguous)
//    Each is an index into the nodes list

// The first 4 items are encoded in disk_metadata
struct disk_metadata {
	size_t num_vecs, dim, num_nodes, num_edges;
};

// Nodes are formatted as follows
struct disk_node {
	size_t data_index; // index back to actual data (return value)
	size_t child_node_index;
	size_t num_edges;
	size_t edges_start_index;
};
template <typename T> struct disk_node_if : disk_node {
	disk_node_if(const disk_node& dn)
			: data_index(dn.data_index), child_node_index(dn.child_node_index),
				num_edges(dn.num_edges), edges_start_index(dn.edges_start_index) {}
	vec<T> v;
};
// followed by an encoding of the corresponding vector

// total size of a node is sizeof(disk_node)+dim*sizeof(T)
//
// total size of nodes array is num_nodes*(sizeof(disk_node)+dim*sizeof(T))
//
// start index of edges array is therefore
// sizeof(disk_metadata)+num_nodes*(sizeof(disk_node)+dim*sizeof(T))
//
// total size is then
// sizeof(disk_metadata)+num_nodes*(sizeof(disk_node)+dim*sizeof(T)) +
// sizeof(size_t)*num_edges

template <typename T> struct disk_index {
	memory_mapped_vector mmv;
	disk_metadata dm;
	disk_index(const std::string& filename, size_t size)
			: mmv(filename.c_str(), size) {
		populate_disk_metadata();
	}
	void populate_disk_metadata() { dm = mmv.read<disk_metadata>(0); }
	const disk_metadata& get_disk_metadata() { return dm; }
	size_t num_vecs() { return get_disk_metadata().num_vecs; }
	size_t dim() { return get_disk_metadata().dim; }
	size_t num_nodes() { return get_disk_metadata().num_nodes; }
	size_t num_edges() { return get_disk_metadata().num_edges; }
	size_t node_array_start() { return sizeof(disk_metadata); }
	size_t node_size() { return sizeof(disk_node) + sizeof(T) * dim(); }
	size_t edge_array_start() {
		return sizeof(disk_metadata) + node_size() * num_nodes();
	}
	size_t edge_size() { return sizeof(size_t); }
	disk_node_if get_node(size_t i) {
		size_t location = node_array_start() + i * node_size();
		disk_node dn = mmv.read<disk_node>(location);
		disk_node_if dnif(dn);
		dnif.v = vec<T>(mmv.read_list<T>(location + sizeof(disk_node), dim()));
		return dnif;
	}
	std::vector<size_t> get_edges(const disk_node& dn) {
		size_t location =
				edge_array_start() + dn.edge_size() * dn.edges_start_index;
		return mmv.read_list<size_t>(location, dn.num_edges);
	}
};

#define NO_CHILD std::numeric_limits<size_t>::max()

template <typename T> struct disk_index_builder {
	// map level -> vertex -> node index
	size_t dim;
	size_t num_vecs;
	std::vector<disk_node> nodes;
	std::vector<size_t> edges;
	const std::vector<vec<T>>& all_entries;
	robin_hood::unordered_flat_map<size_t,
																 robin_hood::unordered_flat_map<size_t, size_t>>
			map_to_node;
	void add_node(size_t level, size_t data_index, const vec<T>& data) {
		map_to_node[level][data_index] = nodes.size();
		disk_node dn(disk_node{data_index, NO_CHILD, 0, 0});
		// dnif.v = data;
		dim = data.dim();
		nodes.emplace_back(dn);
	}
	void add_edges(size_t level, size_t data_index,
								 std::vector<size_t> to_nodes) {
		size_t node_index = map_to_node[level][data_index];
		nodes[node_index].edges_start_index = edges.size();
		nodes[node_index].num_edges = to_nodes.size();
		for (size_t to_node_index : map_to_node[level][to_data_index])
			edges.emplace_back(to_node_index);
	}
	void populate_child_index(size_t level, size_t data_index) {
		size_t node_index = map_to_node[level][data_index];
		size_t child_node_index = map_to_node[level - 1][data_index];
		nodes[node_index].child_node_index = child_node_index;
	}
	disk_index_builder(
			const std::vector<
					robin_hood::unordered_flat_map<size_t, std::vector<size_t>>>& hadj,
			const std::vector<vec<T>>& all_entries) {
		num_vecs = all_entries.size();
		for (size_t level = 0; level < hadj.size(); ++level) {
			for (const auto& [data_index, _] : hadj[level]) {
				add_node(level, data_index, all_entries.at(data_index));
			}
		}
		for (size_t level = 0; level < hadj.size(); ++level) {
			for (const auto& [data_index, edges] : hadj[level]) {
				add_edges(level, data_index, edges);
			}
		}
		for (size_t level = 1; level < hadj.size(); ++level) {
			for (const auto& [data_index, _] : hadj[level]) {
				populate_child_index(level, data_index);
			}
		}
	}
	size_t write(const std::string& filename) { // returns size of file
		disk_metadata dm { num_vecs, dim, nodes.size(), edges.size(); };
		// compute total size
		size_t total_size =
				sizeof(disk_metadata) +
				dm.num_nodes * (sizeof(disk_node) + sizeof(T) * dm.dim) +
				dm.num_edges * sizeof(size_t);
		memory_mapped_vector mmv(filename.c_str(), total_size);
		// write dm, then nodes, then edges
		size_t size_written = 0;
		mmv.write(size_written, dm);
		size_written += sizeof(disk_metadata);
		for (size_t i = 0; i < nodes.size(); ++i) {
			mmv.write(size_written, nodes[i]);
			size_written += sizeof(disk_node);
			std::vector<T> entry = all_entries[i].to_vector();
			mmv.write_list(size_written, entry.begin(), entry.end());
			size_written += entry.size() * sizeof(T);
			assert(entry.size() == dim);
		}
		mmv.write_list(size_written, edges.begin(), edges.end());
		size_written += edges.size() * sizeof(size_t);
		return size_written;
	}
};

template <typename T>
struct disk_ehnsw_engine : public ann_engine<T, disk_ehnsw_engine<T>> {
	std::string filename;
	std::optional<ehnsw_engine_2<T>> builder_engine;
	ehnsw_engine_2_config subconf;
	size_t max_mb_all_accessor;
	size_t max_mb_index_accessor;
	std::optional<disk_index<T>> di;
	disk_ehnsw_engine(disk_ehnsw_engine_config conf)
			: filename(conf.filename),
				builder_engine(std::make_optional<ehnsw_engine_2>(conf.subconf)),
				subconf(conf.subconf), max_mb_all_accessor(conf.max_mb_all_accessor),
				max_mb_index_accessor(conf.max_mb_index_accessor) {}
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<size_t> _query_k_at_layer(const vec<T>& v, size_t k,
																							size_t starting_point,
																							size_t layer);
	const std::vector<std::vector<size_t>>
	_query_k_internal(const vec<T>& v, size_t k, size_t full_search_top_layer);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "Disk EHNSW Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		for (auto& [pname, p] : ehnsw_engine_2(subconf).param_list())
			add_sub_param(pl, "subengine-", pname, p);
		add_param(pl, max_mb_all_accessor);
		add_param(pl, max_mb_index_accessor);
		return pl;
	}
};

template <typename T>
void disk_ehnsw_engine<T>::_store_vector(const vec<T>& v) {
	builder_engine->store_vector(v);
}

template <typename T> void disk_ehnsw_engine<T>::_build() {
	builder_engine->build();

	size_t di_size = 0;
	// flatten the builder_engine graph, store it on disk (+remove the hashmaps)
	{
		disk_index_builder dib(builder_engine->hadj, builder_engine->all_entries);
		di_size = dib.write(filename);
	}

	// Don't allow access to the builder engine after calling build()
	builder_engine.reset();

	// load the disk index that was built
	di = std::make_optional<disk_index>(filename, di_size);
}

// starting point is now a node index
// TODO implement the things below and make the forward declarations match
template <typename T>
const std::vector<size_t>
disk_ehnsw_engine<T>::_query_k_at_layer(const vec<T>& v, size_t k,
																				size_t starting_point) {
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::priority_queue<std::pair<T, size_t>> to_visit;
	robin_hood::unordered_flat_set<size_t> visited;
	auto visit = [&](T d, size_t u) {
		bool is_good =
				!visited.contains(u) && (top_k.size() < k || top_k.top().first > d);
		visited.insert(u);
		if (is_good) {
			top_k.emplace(d, u);		 // top_k is a max heap
			to_visit.emplace(-d, u); // to_visit is a min heap
		}
		if (top_k.size() > k)
			top_k.pop();
		return is_good;
	};
	visit(dist(v, all_entries[starting_point]), starting_point);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		if (top_k.size() == k && -nd > top_k.top().first)
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		for (const auto& u : hadj[layer][cur]) {
			T d_next = dist(v, all_entries[u]);
			visit(d_next, u);
		}
	}
	std::vector<size_t> ret;
	while (!top_k.empty()) {
		ret.push_back(top_k.top().second);
		top_k.pop();
	}
	reverse(ret.begin(), ret.end()); // sort from closest to furthest
	return ret;
}

template <typename T>
const std::vector<std::vector<size_t>>
disk_ehnsw_engine<T>::_query_k_internal(const vec<T>& v, size_t k) {
	auto current = starting_vertex;
	std::priority_queue<std::pair<T, size_t>> top_k;
	std::vector<std::vector<size_t>> ret;
	// for each layer, in decreasing depth
	for (int layer = hadj.size() - 1; layer >= 0; --layer) {
		size_t layer_k = k;
		if (layer > 0)
			layer_k = 1;
		ret.push_back(_query_k_at_layer(v, layer_k, current, layer));
		current = ret.back().front();
	}
	reverse(ret.begin(), ret.end());
	return ret;
}

template <typename T>
std::vector<size_t> disk_ehnsw_engine<T>::_query_k(const vec<T>& v, size_t k) {
	auto ret = _query_k_internal(v, k * num_for_1nn)[0];
	ret.resize(std::min(k, ret.size()));
	return ret;
}
