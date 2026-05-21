#include <torch/extension.h>

#include <vector>

using torch::Tensor;

Tensor moe_packed_forward_cuda(
    const Tensor& h_gate,
    const Tensor& h_up,
    const Tensor& token_indices,
    const Tensor& expert_offsets,
    const Tensor& route_weights,
    const std::vector<Tensor>& gate_payload,
    const std::vector<Tensor>& gate_rank_idx,
    const std::vector<Tensor>& gate_tile_meta,
    const std::vector<Tensor>& gate_slab_meta,
    const std::vector<Tensor>& gate_scale,
    const std::vector<Tensor>& gate_s,
    const std::vector<Tensor>& up_payload,
    const std::vector<Tensor>& up_rank_idx,
    const std::vector<Tensor>& up_tile_meta,
    const std::vector<Tensor>& up_slab_meta,
    const std::vector<Tensor>& up_scale,
    const std::vector<Tensor>& up_s,
    const std::vector<Tensor>& down_payload,
    const std::vector<Tensor>& down_rank_idx,
    const std::vector<Tensor>& down_tile_meta,
    const std::vector<Tensor>& down_slab_meta,
    const std::vector<Tensor>& down_scale,
    const std::vector<Tensor>& down_s,
    int64_t rank_out,
    int64_t intermediate_size,
    int64_t act_type);

static inline void check_cuda_contig(const Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static inline void check_list_len(
    const std::vector<Tensor>& xs,
    int64_t n,
    const char* name) {
    TORCH_CHECK(
        static_cast<int64_t>(xs.size()) == n,
        name,
        " size mismatch, expected ",
        n,
        ", got ",
        xs.size());
}

Tensor moe_packed_forward(
    const Tensor& h_gate,
    const Tensor& h_up,
    const Tensor& token_indices,
    const Tensor& expert_offsets,
    const Tensor& route_weights,
    const std::vector<Tensor>& gate_payload,
    const std::vector<Tensor>& gate_rank_idx,
    const std::vector<Tensor>& gate_tile_meta,
    const std::vector<Tensor>& gate_slab_meta,
    const std::vector<Tensor>& gate_scale,
    const std::vector<Tensor>& gate_s,
    const std::vector<Tensor>& up_payload,
    const std::vector<Tensor>& up_rank_idx,
    const std::vector<Tensor>& up_tile_meta,
    const std::vector<Tensor>& up_slab_meta,
    const std::vector<Tensor>& up_scale,
    const std::vector<Tensor>& up_s,
    const std::vector<Tensor>& down_payload,
    const std::vector<Tensor>& down_rank_idx,
    const std::vector<Tensor>& down_tile_meta,
    const std::vector<Tensor>& down_slab_meta,
    const std::vector<Tensor>& down_scale,
    const std::vector<Tensor>& down_s,
    int64_t rank_out,
    int64_t intermediate_size,
    int64_t act_type) {
    TORCH_CHECK(h_gate.dim() == 2, "h_gate must be 2D [tokens, rank]");
    TORCH_CHECK(h_up.dim() == 2, "h_up must be 2D [tokens, rank]");
    TORCH_CHECK(h_gate.sizes()[0] == h_up.sizes()[0], "h_gate/h_up token dim mismatch");
    TORCH_CHECK(token_indices.dim() == 1, "token_indices must be 1D");
    TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be 1D");
    TORCH_CHECK(route_weights.dim() == 1, "route_weights must be 1D");

    check_cuda_contig(h_gate, "h_gate");
    check_cuda_contig(h_up, "h_up");
    check_cuda_contig(token_indices, "token_indices");
    check_cuda_contig(expert_offsets, "expert_offsets");
    check_cuda_contig(route_weights, "route_weights");

    TORCH_CHECK(h_gate.scalar_type() == torch::kFloat16, "h_gate must be float16");
    TORCH_CHECK(h_up.scalar_type() == torch::kFloat16, "h_up must be float16");
    TORCH_CHECK(token_indices.scalar_type() == torch::kInt, "token_indices must be int32");
    TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt, "expert_offsets must be int32");
    TORCH_CHECK(route_weights.scalar_type() == torch::kFloat, "route_weights must be float32");

    const auto num_experts = expert_offsets.numel() - 1;
    TORCH_CHECK(num_experts >= 0, "expert_offsets must have length >= 1");
    TORCH_CHECK(token_indices.numel() == route_weights.numel(), "token_indices/route_weights size mismatch");

    check_list_len(gate_payload, num_experts, "gate_payload");
    check_list_len(gate_rank_idx, num_experts, "gate_rank_idx");
    check_list_len(gate_tile_meta, num_experts, "gate_tile_meta");
    check_list_len(gate_slab_meta, num_experts, "gate_slab_meta");
    check_list_len(gate_scale, num_experts, "gate_scale");
    check_list_len(gate_s, num_experts, "gate_s");

    check_list_len(up_payload, num_experts, "up_payload");
    check_list_len(up_rank_idx, num_experts, "up_rank_idx");
    check_list_len(up_tile_meta, num_experts, "up_tile_meta");
    check_list_len(up_slab_meta, num_experts, "up_slab_meta");
    check_list_len(up_scale, num_experts, "up_scale");
    check_list_len(up_s, num_experts, "up_s");

    check_list_len(down_payload, num_experts, "down_payload");
    check_list_len(down_rank_idx, num_experts, "down_rank_idx");
    check_list_len(down_tile_meta, num_experts, "down_tile_meta");
    check_list_len(down_slab_meta, num_experts, "down_slab_meta");
    check_list_len(down_scale, num_experts, "down_scale");
    check_list_len(down_s, num_experts, "down_s");

    TORCH_CHECK(intermediate_size > 0, "intermediate_size must be > 0");
    TORCH_CHECK(rank_out > 0, "rank_out must be > 0");
    TORCH_CHECK(act_type == 0, "only SiLU act_type=0 is currently supported");

    return moe_packed_forward_cuda(
        h_gate,
        h_up,
        token_indices,
        expert_offsets,
        route_weights,
        gate_payload,
        gate_rank_idx,
        gate_tile_meta,
        gate_slab_meta,
        gate_scale,
        gate_s,
        up_payload,
        up_rank_idx,
        up_tile_meta,
        up_slab_meta,
        up_scale,
        up_s,
        down_payload,
        down_rank_idx,
        down_tile_meta,
        down_slab_meta,
        down_scale,
        down_s,
        rank_out,
        intermediate_size,
        act_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "BitsMoE packed MoE forward kernels (gate/up/down)";
    m.def(
        "moe_packed_forward",
        &moe_packed_forward,
        "Packed routed-expert forward (CUDA)",
        py::arg("h_gate"),
        py::arg("h_up"),
        py::arg("token_indices"),
        py::arg("expert_offsets"),
        py::arg("route_weights"),
        py::arg("gate_payload"),
        py::arg("gate_rank_idx"),
        py::arg("gate_tile_meta"),
        py::arg("gate_slab_meta"),
        py::arg("gate_scale"),
        py::arg("gate_s"),
        py::arg("up_payload"),
        py::arg("up_rank_idx"),
        py::arg("up_tile_meta"),
        py::arg("up_slab_meta"),
        py::arg("up_scale"),
        py::arg("up_s"),
        py::arg("down_payload"),
        py::arg("down_rank_idx"),
        py::arg("down_tile_meta"),
        py::arg("down_slab_meta"),
        py::arg("down_scale"),
        py::arg("down_s"),
        py::arg("rank_out"),
        py::arg("intermediate_size"),
        py::arg("act_type"));
}
