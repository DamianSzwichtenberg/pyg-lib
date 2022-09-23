#include <ATen/ATen.h>
#include <ATen/native/CPUBlas.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/convert.h"

#include <iostream>

namespace pyg {
namespace ops {

namespace {

void grouped_matmul_out_kernel(const std::vector<at::Tensor>& input,
                               const std::vector<at::Tensor>& other,
                               std::vector<at::Tensor>& out) {
  TORCH_CHECK(input.size() == other.size() && other.size() == out.size(),
              "Size of all input vectors should be equal.");
  for (size_t i = 0; i < out.size(); ++i) {
    TORCH_CHECK(input[i].is_cpu() && other[i].is_cpu() && out[i].is_cpu(),
                "All tensors should be associated with the 'CPU' device.");
  }

  for (size_t i = 0; i < out.size(); ++i)
    at::matmul_out(out[i], input[i], other[i]);
}

std::vector<at::Tensor> grouped_matmul_kernel(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& other) {
  std::vector<at::Tensor> out(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    out[i] = input[i].new_empty({input[i].size(0), other[i].size(-1)});

  grouped_matmul_out_kernel(input, other, out);

  return out;
}

at::Tensor segment_matmul_kernel(const at::Tensor& input,
                                 const at::Tensor& ptr,
                                 const at::Tensor& other) {
  const auto input_contig = input.contiguous();
  const auto other_contig = other.contiguous();
  const auto size = pyg::utils::size_from_ptr(ptr).cpu();
  const auto out = input.new_empty({input.size(0), other.size(-1)});

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "segment_matmul_kernel", [&] {
        const auto input_data = input_contig.data_ptr<scalar_t>();
        const auto other_data = other_contig.data_ptr<scalar_t>();
        const auto size_data = size.data_ptr<int64_t>();
        auto out_data = out.data_ptr<scalar_t>();

        const auto K = input_contig.size(-1);
        const auto N = other_contig.size(-1);
        const auto B = size.size(0);
        std::unordered_map<int64_t, std::vector<int64_t>> groups;
        for (int64_t i = 0; i < B; ++i) {
          const auto M_i = size_data[i];
          if (groups.count(M_i))
            groups[M_i].push_back(i);
          else
            groups.insert({M_i, {i}});
        }

        std::vector<int64_t> m_row_offsets(B + 1, 0);
        std::partial_sum(size_data, size_data + B, m_row_offsets.begin() + 1);

        for (const auto& group_pair : groups) {
          const auto M_i = group_pair.first;
          const auto& indices = group_pair.second;
          const auto bs = indices.size();
          std::vector<const scalar_t*> input_ptrs(bs);
          std::vector<const scalar_t*> other_ptrs(bs);
          std::vector<scalar_t*> out_ptrs(bs);
          for (size_t i = 0; i < bs; ++i) {
            input_ptrs[i] = input_data + (m_row_offsets[indices[i]] * K);
            other_ptrs[i] = other_data + (indices[i] * K * N);
            out_ptrs[i] = out_data + (m_row_offsets[indices[i]] * N);
          }
          const scalar_t** input_ptr = input_ptrs.data();
          const scalar_t** other_ptr = other_ptrs.data();
          scalar_t** out_ptr = out_ptrs.data();
          at::native::cpublas::gemm_batched<scalar_t>(
              at::native::TransposeType::NoTranspose,
              at::native::TransposeType::NoTranspose, bs, M_i, N, K,
              static_cast<scalar_t>(1), input_ptr, M_i, other_ptr, K,
              static_cast<scalar_t>(0), out_ptr, M_i);
        }
      });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::grouped_matmul"),
         TORCH_FN(grouped_matmul_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_matmul"),
         TORCH_FN(segment_matmul_kernel));
}

}  // namespace ops
}  // namespace pyg
