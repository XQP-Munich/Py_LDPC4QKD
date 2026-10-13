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
#include "LDPC4QKD/prebuilt_codes.hpp"


using Idx = std::uint32_t;
using Bit = std::uint8_t;
using BitVec = std::vector<Bit>;
using NumPyArr_u8 = pybind11::array_t<std::uint8_t, pybind11::array::c_style>;
using NumPyArr_double = pybind11::array_t<double, pybind11::array::c_style>;

namespace py = pybind11;

using namespace LDPC4QKD;

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

PYBIND11_MODULE(_core, m) {
    m.doc() = "Pybind11 based wrapper for LDPC4QKD - LDPC Codes for Rate Adaptive Distributed Source Coding.";

    py::enum_<LDPC4QKD::Decoder>(m, "Decoder")
        .value("Flooding", LDPC4QKD::Decoder::Flooding)
        .value("Layered", LDPC4QKD::Decoder::Layered)
        .value("Improved", LDPC4QKD::Decoder::Improved);

    py::class_<RateAdaptiveCode<Idx>>(m, "RateAdaptiveCode")
        .def(py::init<const std::vector<Idx>&, const std::vector<Idx>&>(),
            "Construct from a mother parity check matrix in Compressed Sparse Column (CSC) format, "
            "without rate adaption.",
            py::arg("colptr"), py::arg("rowIdx"))
        .def(py::init<std::vector<std::vector<Idx>> ,
                         std::vector<Idx> ,
                         Idx>(),
            py::arg("mother_pos_varn"), py::arg("rows_to_combine_rate_adapt"),
            py::arg("initial_row_combs") = 0)
        .def("encode_with_ra",
        [](RateAdaptiveCode<Idx> &self,
                const NumPyArr_u8 &input_key,
                std::size_t output_syndrome_length
            ) -> NumPyArr_u8 {
                std::vector<Bit> in_vec = numpy_array_to_std_vector(input_key);
                std::vector<Bit> out_vec;
                self.encode_with_ra(in_vec, out_vec, output_syndrome_length);
                return vector_to_numpy_array(out_vec); // copy result into a numpy array
            },
            "Compute syndrome and rate adapt to specified final syndrome size.",
            py::arg("input_key"), py::arg("output_syndrome_length")
        )
        .def("decode_infer_rate",
            [](RateAdaptiveCode<Idx> &self,
                const NumPyArr_double &llrs,
                const NumPyArr_u8 &syndrome,
                NumPyArr_u8 &out,
                std::size_t max_num_iter,
                double vsat,
                LDPC4QKD::Decoder decoder
            ) -> bool {
                if (static_cast<std::size_t>(out.size()) != self.getNCols()) {
                    throw std::runtime_error(
                        "decode_infer_rate: `out` has wrong size (" + std::to_string(out.size())
                        + "), expected " + std::to_string(self.getNCols())
                        + ".");
                }
                auto llrs_vec = numpy_array_to_std_vector(llrs);
                auto syndrome_vec = numpy_array_to_std_vector(syndrome);
                std::vector<Bit> out_vec;
                bool converged = self.decode_infer_rate<Bit>(
                        llrs_vec, syndrome_vec, out_vec, max_num_iter, vsat, decoder);

                // put the result of `out_vec` into the numpy array `out`.
                // This modifies the Python object at user's side.
                // Python object needs to have correct shape, otherwise `copy_to_numpy_array` throws exception.
                copy_to_numpy_array(out_vec, out);

                return converged;
            }, "Return value says if decoder converged. Modifies parameter `out` to put corrected key."
                "Must provide correct size",
            py::arg("llrs"), py::arg("syndrome"), py::arg("out"),
            py::arg("max_num_iter"), py::arg("vsat"),
            py::arg("decoder") = LDPC4QKD::Decoder::Layered)
        .def("decode_default",
            [](RateAdaptiveCode<Idx> &self,
                const NumPyArr_u8 &noisy_key,
                const NumPyArr_u8 &syndrome,
                NumPyArr_u8 &out,
                double ch_param_estimate
            ) -> bool {
                if (static_cast<std::size_t>(out.size()) != self.getNCols()) {
                    throw std::runtime_error(
                        "decode_default: `out` has wrong size (" + std::to_string(out.size())
                        + "), expected " + std::to_string(self.getNCols())
                        + ". Checked before decoding to avoid wasting time on a doomed call.");
                }
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
               "Must provide correct size.",
            py::arg("noisy_key"), py::arg("syndrome"), py::arg("out"), py::arg("ch_param_estimate"))
        .def("encode_no_ra",
            [](RateAdaptiveCode<Idx> &self,
                const NumPyArr_u8 &input_key
            ) -> NumPyArr_u8 {
                std::vector<Bit> in_vec = numpy_array_to_std_vector(input_key);
                std::vector<Bit> out_vec;
                self.encode_no_ra(in_vec, out_vec);
                return vector_to_numpy_array(out_vec); // copy result into a numpy array
            },
            "Compute syndrome without rate adaption.",
            py::arg("input_key"))
        // functions that get/set parameters of the codec object.
        .def("set_rate", &RateAdaptiveCode<Idx>::set_rate, "single integer argument: number of rate adaption steps",
            py::arg("n_line_combs"))
        .def("encode_at_current_rate",
            [](RateAdaptiveCode<Idx> &self,
                const NumPyArr_u8 &input_key
            ) -> NumPyArr_u8 {
                std::vector<Bit> in_vec = numpy_array_to_std_vector(input_key);
                std::vector<Bit> out_vec;
                self.encode_at_current_rate(in_vec, out_vec);
                return vector_to_numpy_array(out_vec); // copy result into a numpy array
            },
            "Compute syndrome for current rate adaption.",
            py::arg("input_key")
        )
        .def("getPosCheckn", &RateAdaptiveCode<Idx>::getPosCheckn)
        .def("getPosVarn", &RateAdaptiveCode<Idx>::getPosVarn)
        .def("get_n_rows_mother_matrix", &RateAdaptiveCode<Idx>::get_n_rows_mother_matrix)
        .def("get_n_rows_after_rate_adaption", &RateAdaptiveCode<Idx>::get_n_rows_after_rate_adaption)
        .def("getNCols", &RateAdaptiveCode<Idx>::getNCols)
        .def("get_max_ra_steps", &RateAdaptiveCode<Idx>::get_max_ra_steps);

//    m.def("get_input_size", &get_input_size, "Get input size of nth LDPC code");
//    m.def("get_output_size", &get_output_size, "Get output size of nth LDPC code");
    m.def("encode_with", &(encode_with<0, BitVec, BitVec>), "Compute syndrome using nth code",
        py::arg("code_id"), py::arg("key"), py::arg("result"));
    m.def("get_rate_adaptive_code", &HelperFixedSize::get_rate_adaptive_code<Idx>, "Get data specifying ldpc matrix",
        py::arg("id"));

    py::class_<SuitableCodeChoice>(m, "SuitableCodeChoice")
        .def_readonly("code_id", &SuitableCodeChoice::code_id)
        .def_readonly("ldpc_block_size", &SuitableCodeChoice::ldpc_block_size)
        .def_readonly("syndrome_bits_per_block", &SuitableCodeChoice::syndrome_bits_per_block)
        .def_readonly("ecc_type", &SuitableCodeChoice::ecc_type);

    m.def("select_suitable_code", &select_suitable_code,
        "Select a suitable prebuilt code and suggested syndrome size (in bits) for one block of that code, "
        "given an estimated channel parameter and an input block size. Returns None if no code is available "
        "for the given ch_param_estimate, or if the only candidate for it is larger than input_block_size.",
        py::arg("ch_param_estimate"), py::arg("input_block_size"));
}
