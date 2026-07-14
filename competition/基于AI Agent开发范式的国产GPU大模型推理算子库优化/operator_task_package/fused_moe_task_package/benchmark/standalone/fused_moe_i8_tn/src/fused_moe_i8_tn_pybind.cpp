#include <Python.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "fused_moe_i8_tn_runner.h"

namespace py = pybind11;

namespace {

template <typename T>
std::vector<T> copy_1d_array(const py::array &array, const char *name) {
    auto view = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(array);
    if (!view) {
        throw std::invalid_argument(std::string("failed to cast ") + name);
    }
    if (view.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1D array");
    }
    const T *ptr = static_cast<const T *>(view.data());
    return std::vector<T>(ptr, ptr + view.size());
}

std::vector<int8_t> copy_2d_int8_array(const py::array &array, const char *name) {
    auto view = py::array_t<int8_t, py::array::c_style | py::array::forcecast>::ensure(array);
    if (!view) {
        throw std::invalid_argument(std::string("failed to cast ") + name);
    }
    if (view.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D array");
    }
    const int8_t *ptr = static_cast<const int8_t *>(view.data());
    return std::vector<int8_t>(ptr, ptr + view.size());
}

py::array_t<float> run_fused_moe_pybind(const py::array &a,
                                        const py::array &b_col_major,
                                        const py::array &scale_a,
                                        const py::array &scale_b,
                                        const py::array &moe_weights,
                                        const py::array &token_ids,
                                        const py::array &expert_ids,
                                        int topk,
                                        int num_experts = fused_moe_i8_tn::kDefaultNumExperts,
                                        int device_id = 0) {
    auto a_view = py::array_t<int8_t, py::array::c_style | py::array::forcecast>::ensure(a);
    auto b_view = py::array_t<int8_t, py::array::c_style | py::array::forcecast>::ensure(b_col_major);
    if (!a_view || a_view.ndim() != 2) {
        throw std::invalid_argument("a must be a 2D int8 array");
    }
    if (!b_view || b_view.ndim() != 3) {
        throw std::invalid_argument("b_col_major must be a 3D int8 array");
    }

    fused_moe_i8_tn::HostInputs inputs;
    inputs.num_tokens = static_cast<int>(a_view.shape(0));
    inputs.k = static_cast<int>(a_view.shape(1));
    inputs.num_experts = num_experts;
    inputs.n = static_cast<int>(b_view.shape(1));
    inputs.topk = topk;
    inputs.em = inputs.num_tokens * inputs.topk;
    inputs.a = copy_2d_int8_array(a, "a");
    inputs.b_col_major.assign(static_cast<const int8_t *>(b_view.data()),
                              static_cast<const int8_t *>(b_view.data()) + b_view.size());
    inputs.scale_a = copy_1d_array<float>(scale_a, "scale_a");
    inputs.scale_b = copy_1d_array<float>(scale_b, "scale_b");
    inputs.moe_weights = copy_1d_array<float>(moe_weights, "moe_weights");
    inputs.token_ids = copy_1d_array<int>(token_ids, "token_ids");
    inputs.expert_ids = copy_1d_array<int>(expert_ids, "expert_ids");

    fused_moe_i8_tn::RunResult result = fused_moe_i8_tn::run_fused_moe_i8_tn(inputs, device_id);
    std::vector<float> output = fused_moe_i8_tn::bf16_vector_to_float(result.output);

    py::array_t<float> out({result.rows, result.cols});
    std::memcpy(out.mutable_data(), output.data(), output.size() * sizeof(float));
    return out;
}

}  // namespace

PYBIND11_MODULE(fused_moe_i8_tn_pybind, m) {
    m.doc() = "Pybind wrapper for standalone fused_moe_i8_tn";
    m.def("run_fused_moe_i8_tn",
          &run_fused_moe_pybind,
          py::arg("a"),
          py::arg("b_col_major"),
          py::arg("scale_a"),
          py::arg("scale_b"),
          py::arg("moe_weights"),
          py::arg("token_ids"),
          py::arg("expert_ids"),
          py::arg("topk"),
          py::arg("num_experts") = fused_moe_i8_tn::kDefaultNumExperts,
          py::arg("device_id") = 0);
}
