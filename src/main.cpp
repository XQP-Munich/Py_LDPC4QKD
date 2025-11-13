#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>

// standard library
#include <iostream>
#include <string>

// Project-internal sources
#include "LDPC4QKD/rate_adaptive_code.hpp"
#include "LDPC4QKD/encoder_advanced.hpp"

// automatically generated rate adaption arrays
#include "LDPC4QKD/autogen/rate_adaption_2x4_block_4096.hpp"
#include "LDPC4QKD/autogen/rate_adaption_2x4_block_16384.hpp"
#include "LDPC4QKD/autogen/rate_adaption_2x4_block_1048576.hpp"
#include "LDPC4QKD/autogen/rate_adaption_2x6_block_6144.hpp"
#include "LDPC4QKD/autogen/rate_adaption_2x6_block_24576.hpp"
#include "LDPC4QKD/autogen/rate_adaption_2x6_block_1572864.hpp"


using Idx = std::uint32_t;
using Bit = std::uint8_t;
using BitVec = std::vector<Bit>;
using NumPyArr_u8 = pybind11::array_t<std::uint8_t, pybind11::array::c_style>;
using NumPyArr_double = pybind11::array_t<double, pybind11::array::c_style>;

namespace LDPC4QKD {

    template<typename Tout, typename Tin>
    std::vector<std::vector<Tout>> static_cast_vec_vec(std::vector<std::vector<Tin>> const &v) {
        std::vector<std::vector<Tout>> result(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) {
            result[i].reserve(v[i].size());
            for (const auto &val: v[i]) {
                result[i].push_back(val);
            }
        }
        return result;
    }

    template<typename Tout, typename Tin>
    std::vector<Tout> static_cast_vec(std::vector<Tin> const &v) {
        std::vector<Tout> result;
        result.reserve(v.size());
        for (const auto &val: v) {
            result.push_back(val);
        }
        return result;
    }

    template<typename T, std::size_t N>
    std::vector<T> arr_to_vec(std::array<T, N> a) {
        return std::vector<T>(a.begin(), a.end());
    }

    RateAdaptiveCode<Idx> get_rate_adaptive_code(std::size_t id) {
        switch (id) {
            case 0: { // encoder_2048x6144_4663d91
                auto pos_varn = static_cast_vec_vec<Idx>(
                        std::get<0>(LDPC4QKD::all_encoders_tuple).get_pos_varn());
                auto rate_adapt_rows = static_cast_vec<Idx>(
                        arr_to_vec(AutogenRateAdapt_2x6_block_6144::rows));
                return {pos_varn, rate_adapt_rows, 0};
            }
            case 1: { // encoder_8192x24576_71b51c1
                auto pos_varn = static_cast_vec_vec<Idx>(
                        std::get<1>(LDPC4QKD::all_encoders_tuple).get_pos_varn());
                auto rate_adapt_rows = static_cast_vec<Idx>(
                        arr_to_vec(AutogenRateAdapt_2x6_block_24576::rows));
                return {pos_varn, rate_adapt_rows, 0};
            }
            case 2: { // encoder_524288x1572864_4d78a9f
                auto pos_varn = static_cast_vec_vec<Idx>(
                        std::get<2>(LDPC4QKD::all_encoders_tuple).get_pos_varn());
                auto rate_adapt_rows = static_cast_vec<Idx>(
                        arr_to_vec(AutogenRateAdapt_2x6_block_1572864::rows));
                return {pos_varn, rate_adapt_rows, 0};
            }
            case 3: { // encoder_2048x4096_0c809c3
                auto pos_varn = static_cast_vec_vec<Idx>(
                        std::get<3>(LDPC4QKD::all_encoders_tuple).get_pos_varn());
                auto rate_adapt_rows = static_cast_vec<Idx>(
                        arr_to_vec(AutogenRateAdapt_2x4_block_4096::rows));
                return {pos_varn, rate_adapt_rows, 0};
            }
            case 4: { // encoder_8192x16384_3fcad37
                auto pos_varn = static_cast_vec_vec<Idx>(
                        std::get<4>(LDPC4QKD::all_encoders_tuple).get_pos_varn());
                auto rate_adapt_rows = static_cast_vec<Idx>(
                        arr_to_vec(AutogenRateAdapt_2x4_block_16384::rows));
                return {pos_varn, rate_adapt_rows, 0};
            }
            case 5: { // encoder_524288x1048576_9b50f98
                auto pos_varn = static_cast_vec_vec<Idx>(
                        std::get<5>(LDPC4QKD::all_encoders_tuple).get_pos_varn());
                auto rate_adapt_rows = static_cast_vec<Idx>(
                        arr_to_vec(AutogenRateAdapt_2x4_block_1048576::rows));
                return {pos_varn, rate_adapt_rows, 0};
            }
            default: {
                throw std::runtime_error("No code available for requested ID " + std::to_string(id));
            }
        }
    }
}

namespace py = pybind11;

using namespace LDPC4QKD;

py::array_t<double> add_arrays(py::array_t<double> input1, py::array_t<double> input2) {
    py::buffer_info buf1 = input1.request(), buf2 = input2.request();

    if (buf1.ndim != 1 || buf2.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    if (buf1.size != buf2.size)
        throw std::runtime_error("Input shapes must match");

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<double>(buf1.size);

    py::buffer_info buf3 = result.request();

    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *ptr2 = static_cast<double *>(buf2.ptr);
    double *ptr3 = static_cast<double *>(buf3.ptr);

    for (size_t idx = 0; idx < buf1.shape[0]; idx++)
        ptr3[idx] = ptr1[idx] + ptr2[idx];

//    input1[0] = 39;

    return result;
}

/// This makes a copy!
template <typename T>
std::vector<T> numpy_array_to_std_vector(const pybind11::array_t<T, pybind11::array::c_style> &in) {
    py::buffer_info buf_info = in.request();
    if (buf_info.ndim != 1) {
        throw std::runtime_error("Input array must be one-dimensional");
    }
    size_t size = buf_info.size;
    std::vector<T> vec(size);
    std::copy_n(static_cast<T*>(buf_info.ptr), size, vec.data());

    return vec;
}

/// makes a copy!
template <typename T>
py::array_t<T> vector_to_numpy_array(std::vector<T> in) {
    auto size = in.size();
    auto result = py::array_t<T>(size);
    py::buffer_info buf_result = result.request();
    std::copy_n(in.data(), size, static_cast<T*>(buf_result.ptr));

    return result;
}

template <typename T>
void copy_to_numpy_array(const std::vector<T> &in, py::array_t<T, py::array::c_style> &out) {
    auto size = in.size();

    // This gives
    // > ValueError: cannot resize an array that references or is referenced by another array in this way.
    // > Use the np.resize function or refcheck=False
    // And docs for `resize` (in `numpy.h`) say:
    // > resize will succeed only if it makes a reshape, i.e. original size doesn't change
    // So it seems we cannot do this. Expect user to provide correct length!
    // out.resize(py::array::ShapeContainer({size}));

    py::buffer_info buf_info = out.request();
    if (buf_info.ndim != 1) {
        throw std::runtime_error("Input array must be one-dimensional");
    }
    if (buf_info.size != size) {
        throw std::runtime_error("Size mismatch between output size (" + std::to_string(size)
            + ") and size of buffer passed by user (" + std::to_string(buf_info.size) + ").");
    }

    std::copy_n(in.data(), size, static_cast<T*>(buf_info.ptr));
}

auto get_code_small() {
    /// We use this matrix as an example:
    ///    H =  [1 0 1 0 1 0 1
    ///			0 1 1 0 0 1 1
    ///			0 0 0 1 1 1 1]
    ///
    /// To use it, we must convert H to compressed sparse column (CSC) storage:
    std::vector<Idx> colptr{0, 1, 2, 4, 5, 7, 9, 12};
    std::vector<Idx> row_idx{0, 1, 0, 1, 2, 0, 2, 1, 2, 0, 1, 2};
    return LDPC4QKD::RateAdaptiveCode(colptr, row_idx);
}


void modify_array(NumPyArr_u8 &input_array) {
    auto r = input_array.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < r.shape(0); i++) {
        r(i) += 1;
    }
    r(0) = 210;
    copy_to_numpy_array(std::vector<std::uint8_t>{1,2,3}, input_array);
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "Pybind11 based wrapper for LDPC4QKD - LDPC Codes for Rate Adaptive Distributed Source Coding.";

    m.def("get_code_small", &get_code_small, "Get small code for testing"); // TODO remove
    m.def("add_arrays", &add_arrays, "Add two NumPy arrays. just for testing");// TODO remove
    m.def("modify_array", &modify_array, "just for testing");// TODO remove

    py::class_<RateAdaptiveCode<Idx>>(m, "RateAdaptiveCode")
        .def(py::init<std::vector<std::vector<Idx>> ,
                         std::vector<Idx> ,
                         Idx>())
        .def("encode_with_ra",
        [](RateAdaptiveCode<Idx> &self,
                const NumPyArr_u8 &in,
                std::size_t output_syndrome_length
            ) -> NumPyArr_u8 {
                std::vector<Bit> in_vec = numpy_array_to_std_vector(in);
                std::vector<Bit> out_vec;
                self.encode_with_ra(in_vec, out_vec, output_syndrome_length);
                return vector_to_numpy_array(out_vec); // copy result into a numpy array
            },
            "Compute syndrome and rate adapt to specified final syndrome size."
        )
        .def("decode_infer_rate",
            [](RateAdaptiveCode<Idx> &self,
                const NumPyArr_double &llrs,
                const NumPyArr_u8 &syndrome,
                NumPyArr_u8 &out,
                std::size_t max_num_iter,
                double vsat
            ) -> bool {
                auto llrs_vec = numpy_array_to_std_vector(llrs);
                auto syndrome_vec = numpy_array_to_std_vector(syndrome);
                std::vector<Bit> out_vec;
                bool converged = self.decode_infer_rate<Bit>(llrs_vec, syndrome_vec, out_vec, max_num_iter, vsat);

                // put the result of `out_vec` into the numpy array `out`.
                // This modifies the Python object at user's side.
                // Python object needs to have correct shape, otherwise `copy_to_numpy_array` throws exception.
                copy_to_numpy_array(out_vec, out);

                return converged;
            }, "Return value says if decoder converged. Modifies parameter `out` to put corrected key."
                "Must provide correct size")
        .def("decode_default",
            [](RateAdaptiveCode<Idx> &self,
                const NumPyArr_u8 &noisy_key,
                const NumPyArr_u8 &syndrome,
                NumPyArr_u8 &out,
                double ch_param_estimate
            ) -> bool {
                auto llrs_vec = LDPC4QKD::llrs_bsc(numpy_array_to_std_vector(noisy_key), ch_param_estimate);
                auto syndrome_vec = numpy_array_to_std_vector(syndrome);

                std::vector<Bit> out_vec;
                constexpr auto max_num_iterations = 100;
                bool converged = self.decode_infer_rate<Bit>(llrs_vec, syndrome_vec, out_vec, max_num_iterations);

                // put the result of `out_vec` into the numpy array `out`.
                // This modifies the Python object at user's side.
                // Python object needs to have correct shape, otherwise `copy_to_numpy_array` throws exception.
                copy_to_numpy_array(out_vec, out);

                return converged;
            }, "Using default settings. Return value says if decoder converged. Modifies parameter `out`."
               "Must provide correct size.")
        .def("encode_no_ra",
            [](RateAdaptiveCode<Idx> &self,
                const NumPyArr_u8 &in
            ) -> NumPyArr_u8 {
                std::vector<Bit> in_vec = numpy_array_to_std_vector(in);
                std::vector<Bit> out_vec;
                self.encode_no_ra(in_vec, out_vec);
                return vector_to_numpy_array(out_vec); // copy result into a numpy array
            },
            "Compute syndrome without rate adaption.")
        // functions that get/set parameters of the codec object.
        .def("set_rate", &RateAdaptiveCode<Idx>::set_rate, "single integer argument: number of rate adaption steps")
        .def("encode_at_current_rate",
            [](RateAdaptiveCode<Idx> &self,
                const NumPyArr_u8 &in
            ) -> NumPyArr_u8 {
                std::vector<Bit> in_vec = numpy_array_to_std_vector(in);
                std::vector<Bit> out_vec;
                self.encode_at_current_rate(in_vec, out_vec);
                return vector_to_numpy_array(out_vec); // copy result into a numpy array
            },
            "Compute syndrome for current rate adaption."
        )
        .def("getPosCheckn", &RateAdaptiveCode<Idx>::getPosCheckn)
        .def("getPosVarn", &RateAdaptiveCode<Idx>::getPosVarn)
        .def("get_n_rows_mother_matrix", &RateAdaptiveCode<Idx>::get_n_rows_mother_matrix)
        .def("get_n_rows_after_rate_adaption", &RateAdaptiveCode<Idx>::get_n_rows_after_rate_adaption)
        .def("getNCols", &RateAdaptiveCode<Idx>::getNCols)
        .def("get_max_ra_steps", &RateAdaptiveCode<Idx>::get_max_ra_steps);

//    m.def("get_input_size", &get_input_size, "Get input size of nth LDPC code");
//    m.def("get_output_size", &get_output_size, "Get output size of nth LDPC code");
    m.def("encode_with", &(encode_with<0, BitVec, BitVec>), "Compute syndrome using nth code");
    m.def("get_rate_adaptive_code", &get_rate_adaptive_code, "Get data specifying ldpc matrix");
}
