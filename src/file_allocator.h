#pragma once

#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

template <typename T> class file_allocator {
public:
	using value_type = T;

	file_allocator() = default;

	file_allocator(const file_allocator&) noexcept = default;

	file_allocator& operator=(const file_allocator&) noexcept = default;

	using propagate_on_container_copy_assignment = std::true_type;

	template <typename U> file_allocator(const file_allocator<U>&) {}

	T* allocate(std::size_t n) {
		if (n > std::size_t(-1) / sizeof(T))
			throw std::bad_alloc();

		// Creating a unique filename for the temporary file
		std::string filename =
				"/tmp/alloc_" + std::to_string(unique_file_id++) + ".tmp";

		// Open a file
		int fd = open(filename.c_str(), O_RDWR | O_CREAT | O_EXCL, (mode_t)0600);
		if (fd == -1) {
			perror("Error opening file for writing");
			throw std::bad_alloc();
		}

		// Stretch the file size to the size of the memory to be allocated
		if (ftruncate(fd, n * sizeof(T)) == -1) {
			perror("Error resizing file");
			close(fd);
			throw std::bad_alloc();
		}

		// Map the file to memory
		void* addr =
				mmap(0, n * sizeof(T), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (addr == MAP_FAILED) {
			std::cerr << "Error: " << strerror(errno) << std::endl;
			std::cerr << "n=" << n << std::endl;
			std::cerr << "unique_file_id=" << unique_file_id << std::endl;
			perror("Error mapping file to memory");
			close(fd);
			throw std::bad_alloc();
		}

		// Close file descriptor, the mapping stays alive
		close(fd);

		// Remove the file, the space will be freed once the mapping is gone
		unlink(filename.c_str());

		return static_cast<T*>(addr);
	}

	void deallocate(T* p, std::size_t n) noexcept {
		if (p) {
			munmap(p, n * sizeof(T));
		}
	}

	file_allocator select_on_container_copy_construction() const noexcept {
		return file_allocator();
	}

	bool operator==(const file_allocator& other) const noexcept {
		return true; // All instances are interchangeable
	}

	bool operator!=(const file_allocator& other) const noexcept {
		return false; // All instances are interchangeable
	}

	size_t max_size() const noexcept {
		return std::numeric_limits<size_t>::max() / sizeof(T);
	}

private:
	static inline size_t unique_file_id = 0;
};

template <typename T> class zero_allocator {
public:
	using value_type = T;
	using pointer = T*;
	using const_pointer = const T*;
	using void_pointer = void*;
	using const_void_pointer = const void*;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;

	template <class U> struct rebind { using other = zero_allocator<U>; };

	zero_allocator() noexcept = default;
	zero_allocator(const zero_allocator&) noexcept = default;
	zero_allocator& operator=(const zero_allocator&) noexcept = default;

	template <typename U> zero_allocator(const zero_allocator<U>&) noexcept {}

	T* allocate(std::size_t n) {
		if (n > 0) {
			throw std::bad_alloc();
		}
		return nullptr;
	}

	void deallocate(T* p, std::size_t n) noexcept {
		// Nothing to deallocate since we never allocate non-zero memory
	}

	size_type max_size() const noexcept { return 0; }
};
