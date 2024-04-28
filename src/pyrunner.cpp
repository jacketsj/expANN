#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antitopo_engine.h"
#include "vec.h"

#define RECORD_STATS 1

namespace py = pybind11;

namespace MODULE_NAME {
vec<float>::Underlying convert_numpy_to_eigen(py::array_t<float> input_array) {
	py::buffer_info buf_info = input_array.request();
	return Eigen::Map<vec<float>::Underlying>(static_cast<float*>(buf_info.ptr),
																						buf_info.shape[0]);
}
vec<float>::Underlying convert_raw_to_eigen_padded(const float* input_array,
																									 size_t low_size,
																									 size_t high_size) {
	auto mapped_input = Eigen::Map<const Eigen::VectorXf>(input_array, low_size);
	vec<float>::Underlying ret = vec<float>::Underlying::Zero(high_size);
	ret.head(low_size) = mapped_input.head(low_size);
	return ret;
}
vec<float>::Underlying
convert_numpy_to_eigen_padded(py::array_t<float> input_array) {
	py::buffer_info buf_info = input_array.request();
#ifdef DIM
	size_t dimension = DIM;
#else
	size_t dimension = buf_info.shape[0];
#endif
	return convert_raw_to_eigen_padded(static_cast<float*>(buf_info.ptr),
																		 buf_info.shape[0], dimension);
}

PYBIND11_MODULE(MODULE_NAME, m) {
	py::class_<vec<float>>(m, "Vec", py::module_local())
			.def(py::init<>())
			.def(py::init<const std::vector<float>&>())
			.def(py::init([](py::array_t<float> input_array) {
				return vec<float>(convert_numpy_to_eigen_padded(input_array));
			}))
			.def("data",
					 [](vec<float>& v) {
						 return py::memoryview::from_buffer(v.data(), {v.size()},
																								{sizeof(float)});
					 })
			.def("size", &vec<float>::size)
			.def("normalize", &vec<float>::normalize);

	py::class_<antitopo_engine<float>>(m, "AntitopoEngine", py::module_local())
			.def(py::init<size_t, size_t, size_t, size_t, bool>())
			.def("name", &antitopo_engine<float>::name)
			.def("param_list", &antitopo_engine<float>::param_list)
			.def("store_vector", &antitopo_engine<float>::store_vector)
			.def("store_many_vectors",
					 [](antitopo_engine<float>& engine, py::array_t<float> input_array) {
						 py::buffer_info buf_info = input_array.request();
						 if (buf_info.ndim != 2) {
							 throw std::runtime_error("Input should be a 2D NumPy array");
						 }
						 size_t num_vectors = buf_info.shape[0];
						 size_t vector_dim = buf_info.shape[1];
						 float* ptr = static_cast<float*>(buf_info.ptr);
#ifdef DIM
						 size_t dimension = DIM;
#else
						 size_t dimension = vector_dim;
#endif
						 for (int i = 0; i < num_vectors; i++) {
							 engine.store_vector(convert_raw_to_eigen_padded(
									 ptr + i * vector_dim, vector_dim, dimension));
						 }
					 })
			.def("build", &antitopo_engine<float>::build)
			.def("query_k", &antitopo_engine<float>::query_k)
			.def("query_k_numpy",
					 [](antitopo_engine<float>& engine, py::array_t<float> input_array,
							size_t k) {
						 return engine.query_k(
								 vec<float>(convert_numpy_to_eigen_padded(input_array)), k);
					 })
			.def("set_ef_search", &antitopo_engine<float>::set_ef_search);
}
} // namespace MODULE_NAME
