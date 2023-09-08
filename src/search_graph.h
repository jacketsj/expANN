#pragma once

#include <tuple>
#include <vector>

// This is a simple abstraction that can be replaced with something on-disk
template <typename Data> struct DataStore {
	std::vector<Data> internal_data;
	Data get(size_t index) const { return internal_data.at(index); }
	size_t append(const Data& value) {
		size_t ret = internal_data.size();
		internal_data.emplace_back(value);
		return ret;
	}
	void set(size_t index, const Data& value) { internal_data[index] = value; }
};

template <size_t I, typename T, typename VariadicNext>
constexpr size_t index_in_variadic_rec() {
	static_assert(std::is_same<T, VariadicNext>::value,
								"Element not found in Variadic");
	return I;
}
template <size_t I, typename T, typename VariadicNext, typename VariadicRest...>
constexpr size_t index_in_variadic_rec() {
	if constexpr (std::is_same<T, VariadicNext>::value)
		return I;
	else
		return index_in_variadic_rec<I + 1, T, VariadicRest>();
}
template <typename T, typename Variadic...>
constexpr size_t index_in_variadic() {
	return index_in_variadic_rec<0, T, Variadic...>();
}

// simple in-memory implementation
// a DB-based (or specialized disk-based) implementation can be accomplished at
// a later date
template <typename AdditionalDataTypes...> class SearchGraph {
	std::tuple<DataStore<AdditionalDataTypes>...> additional_data_stores;

public:
	struct VIndex {
		// this works for in-memory or static case
		size_t internal_index;
		VIndex(size_t _internal_index) : internal_index(_internal_index) {}
	};

private:
	// in-memory implementation details
	struct Vertex {
		size_t additional_data_type_index;
		size_t additional_data_index;
		std::vector<VIndex> adj_list_out, adj_list_in;
		Vertex(const size_t& _adi) : additional_data_index(_adi) {}
	};
	std::tuple<std::vector<Vertex<AdditionalDataTypes>>...> vertex_list;
	VIndex gen_vindex(const size_t& adi) {
		VIndex ret(vertex_list.size());
		vertex_list.emplace_back(adi);
		return ret;
	}
	Vertex& get_vertex(VIndex vind) { return vertex_list[vind.internal_index]; }

public:
	template <typename AdditionalDataT>
	VIndex add_vertex(const AdditionalDataT& additional_data, size_t max_degree) {
		size_t additional_data_index =
				std::get<AdditionalDataT>(additional_data_stores)
						.append(additional_data);
		VIndex vind = gen_vindex(additional_data_index);
		return vind;
	}
	void add_edge(VIndex from, VIndex to) {
		get_vertex(from).adj_list_out.emplace_back(to);
		get_vertex(to).adj_list_in.emplace_back(from);
	}
	std::vector<VIndex> get_outgoing_edges(VIndex from) {
		return get_vertex(from).adj_list_out;
	}
	std::vector<VIndex> get_incoming_edges(VIndex to) {
		return get_vertex(to).adj_list_in;
	}
	template <typename AdditionalDataT>
	AdditionalDataT get_vertex_data(VIndex vind) const {
		return std::get<AdditionalDataT>(additional_data_stores)
				.get(get_vertex(vind).additional_data_index);
	}
	size_t get_additional_data_type_index(VIndex vind) const {
		return get_vertex(vind).additional_data_type_index;
	}
	template <typename AdditionalDataT>
	void edit_vertex_data(VIndex vind, const AdditionalDataT& new_data) {
		return std::get<AdditionalDataT>(additional_data_stores)
				.set(get_vertex(vind).additional_data_index, new_data);
	}
	template <typename ReturnType>
	ReturnType apply_additional_data_fn(
			VIndex vind,
			std::tuple<std::function<ReturnType(AdditionalDataTypes)>...> fns_tuple) {
		ReturnType ret();
		// For each additional data type:
		// If the data type index is correct, apply the function
		((get_additional_data_type_index(vind) ==
					index_in_variadic<AdditionalDataTypes, AdditionalDataTypes...>() &&
			(ret =
					 std::get<
							 index_in_variadic<AdditionalDataTypes, AdditionalDataTypes...>>(
							 fns_tuple)(get_vertex_data<AdditionalDataTypes>(vind))),
		 ...);
		return ret;
	}
	template <typename ComparisonVal, size_t FilterCount>
	std::array<std::vector<VIndex>, FilterCount> greedy_search(
			std::vector<VIndex> initial_set,
			std::tuple<std::function<ComparisonVal(AdditionalDataTypes)>...>
					objective_functions,
			std::array<std::tuple<std::function<bool(AdditionalDataTypes)>...>,
								 FilterCount>
					filters) {
		// TODO go copy this from another file, but more generalized
		// keep one to_visit topk for each filter
		// should output one answer for each possible filter
		//
		// when using objective functions or filters,
		// pass them through apply_additional_data_fn
	}
};
