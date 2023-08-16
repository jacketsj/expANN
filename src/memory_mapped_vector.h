#pragma once

#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

class memory_mapped_vector {
public:
	memory_mapped_vector(const char* filename, size_t size,
											 bool read_only = false)
			: size_(size), data_(nullptr) {
		fd_ = open(filename, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
		if (fd_ == -1) {
			throw std::runtime_error("Error opening file");
		}

		if (!read_only) {
			// Write a byte to the end of the file so that it exists and has the
			// correct size
			if (lseek(fd_, size_ - 1, SEEK_SET) == -1) {
				close(fd_);
				throw std::runtime_error("Error seeking file");
			}
			if (write(fd_, "", 1) == -1) {
				close(fd_);
				throw std::runtime_error("Error writing to file");
			}
		}
		// Reset the file pointer to the beginning of the file
		if (lseek(fd_, 0, SEEK_SET) == -1) {
			close(fd_);
			throw std::runtime_error("Error seeking file");
		}

		std::cerr << "mmapping size=" << size_ << std::endl;

		auto prot = PROT_READ;
		if (!read_only)
			prot |= PROT_WRITE;
		// data_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
		data_ = mmap(nullptr, size_, prot, MAP_SHARED, fd_, 0);
		if (data_ == MAP_FAILED) {
			close(fd_);
			throw std::runtime_error("Error mapping file to memory");
		}
	}

	~memory_mapped_vector() {
		if (data_ != nullptr) {
			munmap(data_, size_);
		}
		if (fd_ != -1) {
			close(fd_);
		}
	}

	template <typename T> T tread(size_t offset) const {
		if (offset + sizeof(T) > size_) {
			throw std::out_of_range("Reading out of bounds");
		}
		return *reinterpret_cast<T*>(static_cast<char*>(data_) + offset);
		// T value;
		// std::memcpy(&value, static_cast<char*>(data_) + offset, sizeof(T));
		// return value;
	}

	template <typename T> void twrite(size_t offset, const T& value) {
		if (offset + sizeof(T) > size_) {
			throw std::out_of_range("Writing out of bounds");
		}
		std::memcpy(static_cast<char*>(data_) + offset, &value, sizeof(T));
	}

	template <typename T, typename InputIterator>
	void write_list(size_t offset, InputIterator begin, InputIterator end) {
		size_t count = std::distance(begin, end);
		if (offset + count * sizeof(T) > size_) {
			throw std::out_of_range("Writing out of bounds");
		}
		std::memcpy(static_cast<char*>(data_) + offset, &(*begin),
								count * sizeof(T));
	}

	template <typename T>
	std::vector<T> read_list(size_t offset, size_t count) const {
		if (offset + count * sizeof(T) > size_) {
			throw std::out_of_range("Reading out of bounds");
		}
		std::vector<T> buffer(count);
		std::memcpy(buffer.data(), static_cast<char*>(data_) + offset,
								count * sizeof(T));
		return buffer;
	}

	// asynchronous pre-fetch hint
	void hint(size_t offset, size_t length) {
		if (offset + length > size_) {
			throw std::out_of_range("Hinting out of bounds");
		}
		std::thread prefetchThread([this, offset, length]() {
			madvise(static_cast<char*>(data_) + offset, length, MADV_WILLNEED);
		});
		prefetchThread.detach();
	}

private:
	int fd_;
	size_t size_;
	void* data_;
};
