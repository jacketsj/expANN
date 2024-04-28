#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antitopo_engine.h"
#include "vec.h"

namespace py = pybind11;

namespace MODULE_NAME {
vec<float>::Underlying convert_numpy_to_eigen(py::array_t<float> input_array) {
	py::buffer_info buf_info = input_array.request();
	return Eigen::Map<vec<float>::Underlying>(static_cast<float*>(buf_info.ptr),
																						buf_info.shape[0]);
}
vec<float>::Underlying
convert_numpy_to_eigen_padded(py::array_t<float> input_array) {
	py::buffer_info buf_info = input_array.request();
#ifdef DIM
	size_t dimension = DIM;
#else
	size_t dimension = buf_info.shape[0];
#endif
	auto mapped_input = Eigen::Map<const Eigen::VectorXf>(
			static_cast<float*>(buf_info.ptr), buf_info.shape[0]);
	vec<float>::Underlying ret = vec<float>::Underlying::Zero(dimension);
	ret.head(buf_info.shape[0]) = mapped_input.head(buf_info.shape[0]);
	return ret;
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
			.def("build", &antitopo_engine<float>::build)
			.def("query_k", &antitopo_engine<float>::query_k)
			.def("set_ef_search", &antitopo_engine<float>::set_ef_search);
}
} // namespace MODULE_NAME
