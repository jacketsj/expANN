#pragma once

#include <algorithm>
#include <queue>

#include "robin_hood.h"
// TODO small size optimization

template <typename T>
struct topk_t : private std::priority_queue<std::pair<T, size_t>> {

private:
	typedef std::pair<T, size_t> dat;
	using std::priority_queue<dat>::top;
	using std::priority_queue<dat>::pop;
	using std::priority_queue<dat>::emplace;
	using std::priority_queue<dat>::empty;
	robin_hood::unordered_flat_set<size_t> known;

public:
	using std::priority_queue<dat>::size;
	size_t k;
	topk_t(size_t _k) : k(_k) {
		// std::cout << "New topk_t" << std::endl;
	}
	bool consider(const T& d, size_t v) {
		// std::cout << "considering v=" << v << std::endl;
		bool is_good = !known.contains(v) && (size() < k || top().first > d);
		if (is_good) {
			emplace(d, v); // max heap
			known.emplace(v);
			// std::cout << "adding v=" << v << std::endl;
		}
		if (size() > k) {
			// std::cout << "removing oldv=" << top().second << std::endl;
			known.erase(top().second);
			pop();
		}
		return is_good;
	}
	void discard_until_size(size_t goal) {
		while (size() > goal) {
			known.erase(top().second);
			pop();
		}
	}
	size_t worst() const { return top().second; }
	bool at_capacity() const { return size() == k; }
	const T& worst_val() const { return top().first; }
	std::vector<size_t> to_vector() const {
		std::vector<size_t> ret;
		std::priority_queue<dat> dupe(
				static_cast<const std::priority_queue<dat>&>(*this));
		while (!dupe.empty()) {
			ret.push_back(dupe.top().second);
			dupe.pop();
		}
		std::reverse(ret.begin(), ret.end()); // sort from closest to furthest
		return ret;
	}
};
