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
#include "mio.hpp"
#include "robin_hood.h"

struct disk_ehnsw_engine_config {
	ehnsw_engine_2_config subconf;
	std::string filename;
	disk_ehnsw_engine_config(ehnsw_engine_2_config _subconf,
													 const std::string& _filename)
			: subconf(_subconf), filename(_filename) {}
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
	// size_t num_vecs, dim, num_nodes, num_edges, starting_node;
	size_t dim, num_nodes, num_edges, starting_node;
};

// Nodes are formatted as follows
struct disk_node {
	size_t data_index; // index back to actual data (return value)
	size_t child_node_index;
	size_t num_edges;
	size_t edges_start_index;
	// disk_node(size_t _data_index, size_t _child_node_index, size_t _num_edges,
	//					size_t _edges_start_index)
	//		: data_index(_data_index), child_node_index(_child_node_index),
	//			num_edges(_num_edges), edges_start_index(_edges_start_index) {}
};
template <typename T> struct disk_node_if : disk_node {
	vec<T> v;
	disk_node_if(const disk_node& dn) : disk_node(dn) {}
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

template <typename T>
void mio_read(mio::mmap_source& miom, T& destination, size_t offset,
							size_t size) {
	std::memcpy((char*)(&destination), miom.data() + offset, size);
}

template <typename T>
void mio_write(mio::mmap_sink& miom, T& source, size_t offset, size_t size) {
	std::memcpy(miom.data<mio::access_mode::write>() + offset, (char*)(&source),
							size);
}

template <typename T> struct disk_index {
	mio::mmap_source miom;
	// memory_mapped_vector mmv;
	disk_metadata dm;
	disk_index(const std::string& filename, size_t size)
			: miom(filename.c_str()) {
		// , mmv(filename.c_str(), size, true)
		populate_disk_metadata();
	}
	void populate_disk_metadata() {
		// for (size_t i = 0; i < sizeof(disk_metadata); ++i) {
		//	*((char*)(&dm) + i) = miom[i];
		// }
		mio_read(miom, dm, 0, sizeof(disk_metadata));
		// mio_write(miom, dm, 0, sizeof(disk_metadata));
		// dm = mmv.tread<disk_metadata>(0);
	}
	const disk_metadata& get_disk_metadata() { return dm; }
	// size_t num_vecs() { return get_disk_metadata().num_vecs; }
	size_t dim() { return get_disk_metadata().dim; }
	size_t num_nodes() { return get_disk_metadata().num_nodes; }
	size_t num_edges() { return get_disk_metadata().num_edges; }
	size_t node_array_start() { return sizeof(disk_metadata); }
	size_t node_size() { return sizeof(disk_node) + sizeof(T) * dim(); }
	size_t edge_array_start() {
		return sizeof(disk_metadata) + node_size() * num_nodes();
	}
	size_t edge_size() { return sizeof(size_t); }
	disk_node_if<T> get_node(size_t i) {
		size_t location = node_array_start() + i * node_size();
		// std::cerr << "Reading i=" << i << "=location=" << location << std::endl;
		// std::cerr << "dm.dim=" << dm.dim << std::endl;
		//  disk_node dn = mmv.tread<disk_node>(location);
		disk_node dn{0, 0, 0, 0};
		// for (size_t i = 0; i < sizeof(disk_node); ++i) {
		//	*((char*)(&dn) + i) = miom[location + i];
		// }
		mio_read(miom, dn, location, sizeof(disk_node));
		// std::cerr << "Completed read, dn.data_index=" << dn.data_index <<
		// std::endl; std::cerr << "Completed read, dn.child_node_index=" <<
		// dn.child_node_index
		//					<< std::endl;
		// std::cerr << "Completed read, dn.num_edges=" << dn.num_edges <<
		// std::endl; std::cerr << "Completed read, dn.edges_start_index=" <<
		// dn.edges_start_index
		//					<< std::endl;
		//   dn.data_index = mmv.tread<size_t>(location);
		//   dn.child_node_index = mmv.tread<size_t>(location + sizeof(size_t));
		//    dn.num_edges = mmv.tread<size_t>(location + sizeof(size_t) * 2);
		//    dn.edges_start_index = mmv.tread<size_t>(location + sizeof(size_t) *
		//    3);
		disk_node_if<T> dnif(dn);
		// dnif.v.set_dim(dim());
		std::vector<T> v_data(dim());
		// for (size_t i = 0; i < sizeof(T) * dim(); ++i) {
		//	*((char*)(v_data.data()) + i) = miom[location + disk_node + i];
		// }
		mio_read(miom, *v_data.data(), location + sizeof(disk_node),
						 sizeof(T) * dim());
		dnif.v = vec<T>(v_data);
		// dnif.v = vec<T>(mmv.read_list<T>(location + sizeof(disk_node), dim()));
		// std::cerr << "Completed read of dnif" << std::endl;
		return dnif;
	}
	size_t get_starting_node_index() { return dm.starting_node; }
	std::vector<size_t> get_edges(const disk_node& dn) {
		size_t location =
				edge_array_start() + sizeof(size_t) * dn.edges_start_index;
		std::vector<size_t> ret(dn.num_edges);
		for (size_t i = 0; i < sizeof(size_t) * dn.num_edges; ++i) {
			*((char*)(ret.data()) + i) = miom[location + i];
		}
		// std::cerr << "Neighbours: ";
		// for (size_t k : ret)
		// std::cerr << k << ',';
		// std::cerr << std::endl;
		return ret;
		// return mmv.read_list<size_t>(location, dn.num_edges);
	}
};

#define NO_CHILD std::numeric_limits<size_t>::max()

template <typename T> struct disk_index_builder {
	// map level -> vertex -> node index
	size_t dim;
	// size_t num_vecs;
	size_t starting_node;
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
		for (size_t to_data_index : to_nodes)
			edges.emplace_back(map_to_node[level][to_data_index]);
	}
	void populate_child_index(size_t level, size_t data_index) {
		size_t node_index = map_to_node[level][data_index];
		size_t child_node_index = map_to_node[level - 1][data_index];
		nodes[node_index].child_node_index = child_node_index;
	}
	disk_index_builder(
			const std::vector<
					robin_hood::unordered_flat_map<size_t, std::vector<size_t>>>& hadj,
			const std::vector<vec<T>>& _all_entries, size_t starting_vertex)
			: all_entries(_all_entries) {
		// num_vecs = all_entries.size();
		for (size_t level = 0; level < hadj.size(); ++level) {
			for (const auto& [data_index, _] : hadj[level]) {
				add_node(level, data_index, all_entries.at(data_index));
			}
		}
		starting_node = map_to_node[hadj.size() - 1][starting_vertex];
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
		// disk_metadata dm{num_vecs, dim, nodes.size(), edges.size(),
		// starting_node};
		disk_metadata dm{dim, nodes.size(), edges.size(), starting_node};
		// TODO this might rely on dim being a multiple of 2 due to alignment
		// issues. Test it (and round up if necessary)

		// std::cerr << "num_vecs=" << dm.num_vecs << std::endl;
		// std::cerr << "num_nodes=" << dm.num_nodes << std::endl;
		// std::cerr << "num_edges=" << dm.num_edges << std::endl;
		// std::cerr << "dim=" << dm.dim << std::endl;
		// std::cerr << "starting_node=" << dm.starting_node << std::endl;

		// compute total size
		size_t total_size =
				sizeof(disk_metadata) +
				dm.num_nodes * (sizeof(disk_node) + sizeof(T) * dm.dim) +
				dm.num_edges * sizeof(size_t);

		// std::cerr << "total_size=" << total_size << " bytes" << std::endl;

		// allocate the file
		{
			std::ofstream file(filename.c_str());
			std::string s(total_size, '0');
			file << s;
		}
		// use mio mmap to write to it
		mio::mmap_sink miom(filename.c_str());
		// memory_mapped_vector mmv(filename.c_str(), total_size);
		// write dm, then nodes, then edges
		size_t size_written = 0;
		mio_write(miom, dm, size_written, sizeof(disk_metadata));
		// mmv.twrite(size_written, dm);
		size_written += sizeof(disk_metadata);
		for (size_t i = 0; i < nodes.size(); ++i) {
			// std::cerr << "About to write node i=" << i
			// 					<< " at location=" << size_written
			// 					<< "(actual node data index=" << nodes[i].data_index << ")"
			// 					<< std::endl;
			mio_write(miom, nodes[i], size_written, sizeof(disk_node));
			// mmv.twrite(size_written, nodes[i]);
			size_written += sizeof(disk_node);
			std::vector<T> entry = all_entries[nodes[i].data_index].to_vector();
			mio_write(miom, *entry.data(), size_written, sizeof(T) * entry.size());
			// mmv.write_list<T>(size_written, entry.begin(), entry.end());
			size_written += entry.size() * sizeof(T);
			assert(entry.size() == dim);
		}
		mio_write(miom, *edges.data(), size_written, sizeof(size_t) * edges.size());
		// mmv.write_list<size_t>(size_written, edges.begin(), edges.end());
		size_written += edges.size() * sizeof(size_t);
		// std::cerr << "size_written=" << size_written << std::endl;
		return size_written;
	}
};

template <typename T>
struct disk_ehnsw_engine : public ann_engine<T, disk_ehnsw_engine<T>> {
	std::string filename;
	std::optional<ehnsw_engine_2<T>> builder_engine;
	ehnsw_engine_2_config subconf;
	std::optional<disk_index<T>> di;
	size_t num_for_1nn;
	disk_ehnsw_engine(disk_ehnsw_engine_config conf)
			: filename(conf.filename),
				builder_engine(std::make_optional<ehnsw_engine_2<T>>(conf.subconf)),
				subconf(conf.subconf), num_for_1nn(conf.subconf.num_for_1nn) {}
	void _store_vector(const vec<T>& v);
	void _build();
	const std::vector<size_t> _query_k_at_layer(const vec<T>& v, size_t k,
																							size_t starting_point);
	const std::vector<std::vector<size_t>> _query_k_internal(const vec<T>& v,
																													 size_t k);
	std::vector<size_t> _query_k(const vec<T>& v, size_t k);
	const std::string _name() { return "Disk EHNSW Engine"; }
	const param_list_t _param_list() {
		param_list_t pl;
		for (auto& [pname, p] : ehnsw_engine_2<T>(subconf).param_list())
			add_sub_param(pl, "subengine-", pname, p);
		// add_param(pl, filename.c_str());
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
		// std::cerr << "Finished building, now flattening." << std::endl;
		disk_index_builder<T> dib(builder_engine->hadj, builder_engine->all_entries,
															builder_engine->starting_vertex);
		// std::cerr << "Finished flatterning, now writing." << std::endl;
		di_size = dib.write(filename);
		// di_size = 6960;
		// std::cerr << "Finished writing." << std::endl;
	}

	// Don't allow access to the builder engine after calling build()
	builder_engine.reset();

	// load the disk index that was built
	di = std::make_optional<disk_index<T>>(filename, di_size);
}

// starting point is now a node index
template <typename T>
const std::vector<size_t>
disk_ehnsw_engine<T>::_query_k_at_layer(const vec<T>& v, size_t k,
																				size_t starting_point) {
	std::priority_queue<std::pair<T, size_t>> top_k; // stores distance + node_id
	std::priority_queue<std::pair<T, size_t>>
			to_visit;																		// stores distance + node_id
	robin_hood::unordered_flat_set<size_t> visited; // stores node_ids
	auto visit = [&](T d, size_t u) {								// takes distance + node_id
		bool is_good =
				!visited.contains(u) && (top_k.size() < k || top_k.top().first > d);
		visited.insert(u);
		if (is_good) {
			// TODO pre-fetch/hint edges here maybe
			top_k.emplace(d, u);		 // top_k is a max heap
			to_visit.emplace(-d, u); // to_visit is a min heap
		}
		if (top_k.size() > k)
			top_k.pop();
		return is_good;
	};
	visit(dist2(v, di->get_node(starting_point).v), starting_point);
	while (!to_visit.empty()) {
		T nd;
		size_t cur;
		std::tie(nd, cur) = to_visit.top();
		if (top_k.size() == k && -nd > top_k.top().first)
			// everything neighbouring current best set is already evaluated
			break;
		to_visit.pop();
		for (const auto& u : di->get_edges(di->get_node(cur))) {
			T d_next = dist2(v, di->get_node(u).v);
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
	auto current = di->get_starting_node_index();
	std::vector<std::vector<size_t>> ret;
	// for each layer, in decreasing depth
	while (current != NO_CHILD) {
		size_t layer_k = k;
		if (di->get_node(current).child_node_index != NO_CHILD)
			layer_k = 1;
		ret.push_back(_query_k_at_layer(v, layer_k, current));
		current = di->get_node(ret.back().front()).child_node_index;
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
