#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

struct VisitedContainer {
	bool taken;
	std::vector<char> visited; // booleans
	std::vector<size_t> visited_recent;
	VisitedContainer(size_t data_size)
			: taken(false), visited(data_size), visited_recent() {}
};
struct VisitedContainerRef {
	std::mutex& visited_containers_mutex;
	size_t index;
	bool thread_safe;
	std::shared_ptr<VisitedContainer> vref;
	VisitedContainerRef(std::shared_ptr<VisitedContainer> visited_container_ref,
											std::mutex& visited_containers_mutex, size_t index,
											size_t data_size, bool thread_safe = false)
			: visited_containers_mutex(visited_containers_mutex), index(index),
				thread_safe(thread_safe), vref(visited_container_ref) {
		if (vref->taken == true) {
			throw std::runtime_error(
					"Overlapping refs detected! Not thread safe! index=" +
					std::to_string(index));
		}
		if (vref->visited.size() < data_size) {
			vref->visited.resize(data_size);
		}
		vref->taken = true;
	}
	inline bool Visit(const size_t& data_index) {
		if (vref->visited[data_index]) {
			return false;
		}
		vref->visited[data_index] = true;
		vref->visited_recent.emplace_back(data_index);
		return true;
	}
	~VisitedContainerRef() {
		for (const size_t& data_index : vref->visited_recent) {
			vref->visited[data_index] = false;
		}
		vref->visited_recent.clear();
		if (!thread_safe) {
			std::lock_guard<std::mutex> lock(visited_containers_mutex);
			vref->taken = false;
		} else {
			vref->taken = false;
		}
	}
};
struct VisitedContainerManager {
	std::vector<std::shared_ptr<VisitedContainer>> visited_containers;
	std::mutex visited_containers_mutex;
	void resize_visit_containers(size_t num_threads_lower_bound,
															 size_t data_size) {
		if (visited_containers.size() < num_threads_lower_bound) {
			std::lock_guard<std::mutex> lock(visited_containers_mutex);
			while (visited_containers.size() < num_threads_lower_bound) {
				visited_containers.emplace_back(
						std::make_shared<VisitedContainer>(data_size));
			}
		}
	}
	VisitedContainerRef get_visitref(std::optional<size_t> thread_index,
																	 size_t data_size) {
		if (thread_index.has_value()) {
			size_t ti = thread_index.value() + 1;
			resize_visit_containers(ti, data_size);
			return VisitedContainerRef(visited_containers[thread_index.value()],
																 visited_containers_mutex, thread_index.value(),
																 data_size, true);
		}
		std::lock_guard<std::mutex> lock(visited_containers_mutex);
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
		return VisitedContainerRef(visited_containers[index],
															 visited_containers_mutex, index, data_size);
	}
};
