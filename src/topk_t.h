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
	topk_t(size_t _k) : k(_k) {}
	bool consider(const T& d, size_t v) {
		bool is_good = !known.contains(v) && (size() < k || top().first > d);
		if (is_good) {
			emplace(d, v); // max heap
			known.emplace(v);
		}
		if (size() > k) {
			known.erase(top().second);
			pop();
		}
		return is_good;
	}
	void discard_until_size(size_t goal) {
		while (size() > goal)
			pop();
	}
	size_t worst() const { return top().second; }
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
