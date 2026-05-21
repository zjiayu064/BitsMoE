#include <torch/extension.h>

torch::Tensor pack_1bit_cuda(const torch::Tensor& input);
torch::Tensor unpack_1bit_cuda(const torch::Tensor& input);
torch::Tensor pack_int8_1bit_cuda(const torch::Tensor& input);
torch::Tensor unpack_int8_1bit_cuda(const torch::Tensor& input);

torch::Tensor pack_2bit_cuda(const torch::Tensor& input);
torch::Tensor unpack_2bit_cuda(const torch::Tensor& input);
torch::Tensor pack_int8_2bit_cuda(const torch::Tensor& input);
torch::Tensor unpack_int8_2bit_cuda(const torch::Tensor& input);

torch::Tensor pack_3bit_cuda(const torch::Tensor& input);
torch::Tensor unpack_3bit_cuda(const torch::Tensor& input);
torch::Tensor pack_int8_3bit_cuda(const torch::Tensor& input);
torch::Tensor unpack_int8_3bit_cuda(const torch::Tensor& input);

torch::Tensor pack_4bit_cuda(const torch::Tensor& input);
torch::Tensor unpack_4bit_cuda(const torch::Tensor& input);
torch::Tensor pack_int8_4bit_cuda(const torch::Tensor& input);
torch::Tensor unpack_int8_4bit_cuda(const torch::Tensor& input);

torch::Tensor pack_8bit_cuda(const torch::Tensor& input);
torch::Tensor unpack_8bit_cuda(const torch::Tensor& input);
torch::Tensor pack_int8_8bit_cuda(const torch::Tensor& input);
torch::Tensor unpack_int8_8bit_cuda(const torch::Tensor& input);

torch::Tensor pack_6bit_cuda(const torch::Tensor& input);
torch::Tensor unpack_6bit_cuda(const torch::Tensor& input);
torch::Tensor pack_int8_6bit_cuda(const torch::Tensor& input);
torch::Tensor unpack_int8_6bit_cuda(const torch::Tensor& input);

torch::Tensor int8_to_uint8_cuda(const torch::Tensor& input, int offset);
torch::Tensor uint8_to_int8_cuda(const torch::Tensor& input, int offset);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_1bit", &pack_1bit_cuda, "Pack 1-bit (CUDA)");
    m.def("unpack_1bit", &unpack_1bit_cuda, "Unpack 1-bit (CUDA)");
    m.def("pack_int8_1bit", &pack_int8_1bit_cuda, "Pack int8 1-bit (CUDA)");
    m.def("unpack_int8_1bit", &unpack_int8_1bit_cuda, "Unpack int8 1-bit (CUDA)");

    m.def("pack_2bit", &pack_2bit_cuda, "Pack 2-bit (CUDA)");
    m.def("unpack_2bit", &unpack_2bit_cuda, "Unpack 2-bit (CUDA)");
    m.def("pack_int8_2bit", &pack_int8_2bit_cuda, "Pack int8 2-bit (CUDA)");
    m.def("unpack_int8_2bit", &unpack_int8_2bit_cuda, "Unpack int8 2-bit (CUDA)");

    m.def("pack_3bit", &pack_3bit_cuda, "Pack 3-bit (CUDA)");
    m.def("unpack_3bit", &unpack_3bit_cuda, "Unpack 3-bit (CUDA)");
    m.def("pack_int8_3bit", &pack_int8_3bit_cuda, "Pack int8 3-bit (CUDA)");
    m.def("unpack_int8_3bit", &unpack_int8_3bit_cuda, "Unpack int8 3-bit (CUDA)");

    m.def("pack_4bit", &pack_4bit_cuda, "Pack 4-bit (CUDA)");
    m.def("unpack_4bit", &unpack_4bit_cuda, "Unpack 4-bit (CUDA)");
    m.def("pack_int8_4bit", &pack_int8_4bit_cuda, "Pack int8 4-bit (CUDA)");
    m.def("unpack_int8_4bit", &unpack_int8_4bit_cuda, "Unpack int8 4-bit (CUDA)");

    m.def("pack_8bit", &pack_8bit_cuda, "Pack 8-bit (CUDA)");
    m.def("unpack_8bit", &unpack_8bit_cuda, "Unpack 8-bit (CUDA)");
    m.def("pack_int8_8bit", &pack_int8_8bit_cuda, "Pack int8 8-bit (CUDA)");
    m.def("unpack_int8_8bit", &unpack_int8_8bit_cuda, "Unpack int8 8-bit (CUDA)");

    m.def("pack_6bit", &pack_6bit_cuda, "Pack 6-bit (CUDA)");
    m.def("unpack_6bit", &unpack_6bit_cuda, "Unpack 6-bit (CUDA)");
    m.def("pack_int8_6bit", &pack_int8_6bit_cuda, "Pack int8 6-bit (CUDA)");
    m.def("unpack_int8_6bit", &unpack_int8_6bit_cuda, "Unpack int8 6-bit (CUDA)");

    m.def("int8_to_uint8", &int8_to_uint8_cuda, "int8->uint8 (with offset, CUDA)");
    m.def("uint8_to_int8", &uint8_to_int8_cuda, "uint8->int8 (with offset, CUDA)");
}
