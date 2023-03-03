#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

enum FnType { ADD, SUB, MUL, DIV };
const std::map<std::string, FnType> to_fn_type = {
    {"add", ADD},
    {"sub", SUB},
    {"mul", MUL},
    {"div", DIV},
};

template <typename scalar_t>
void inner_loop_add(const scalar_t* right,
                    const scalar_t* left,
                    scalar_t* out,
                    int64_t num_features) {
  for (size_t i = 0; i < num_features; ++i) {
    out[i] = left[i] + right[i];
  }
}

template <typename scalar_t>
void inner_loop_sub(const scalar_t* right,
                    const scalar_t* left,
                    scalar_t* out,
                    int64_t num_features) {
  for (size_t i = 0; i < num_features; ++i) {
    out[i] = left[i] - right[i];
  }
}

template <typename scalar_t>
void inner_loop_mul(const scalar_t* right,
                    const scalar_t* left,
                    scalar_t* out,
                    int64_t num_features) {
  for (size_t i = 0; i < num_features; ++i) {
    out[i] = left[i] * right[i];
  }
}

template <typename scalar_t>
void inner_loop_div(const scalar_t* right,
                    const scalar_t* left,
                    scalar_t* out,
                    int64_t num_features) {
  for (size_t i = 0; i < num_features; ++i) {
    out[i] = left[i] / right[i];
  }
}

at::Tensor sampled_op_kernel(const at::Tensor& left,
                             const at::Tensor& right,
                             const at::optional<at::Tensor> left_index,
                             const at::optional<at::Tensor> right_index,
                             const std::string fn) {
  const auto index_val_options = at::TensorOptions().dtype(at::kLong);
  const auto left_index_val =
      left_index.value_or(at::arange(left.size(0), index_val_options));
  const auto right_index_val =
      right_index.value_or(at::arange(right.size(0), index_val_options));
  const auto num_edges = left_index_val.size(0);
  const auto num_features = left.size(-1);
  auto out = left.new_empty({num_edges, num_features});
  AT_DISPATCH_ALL_TYPES(left.scalar_type(), "sampled_op_kernel", [&] {
    using inner_loop_ptr =
        void (*)(const scalar_t*, const scalar_t*, scalar_t*, int64_t);
    inner_loop_ptr inner_loop = nullptr;
    auto fn_type = to_fn_type.at(fn);
    if (fn_type == ADD) {
      inner_loop = inner_loop_add<scalar_t>;
    } else if (fn_type == SUB) {
      inner_loop = inner_loop_sub<scalar_t>;
    } else if (fn_type == MUL) {
      inner_loop = inner_loop_mul<scalar_t>;
    } else {
      inner_loop = inner_loop_div<scalar_t>;
    }
    const auto left_index_base_ptr = left_index_val.data_ptr<int64_t>();
    const auto right_index_base_ptr = right_index_val.data_ptr<int64_t>();
    const auto left_base_ptr = left.data_ptr<scalar_t>();
    const auto right_base_ptr = right.data_ptr<scalar_t>();
    auto out_base_ptr = out.data_ptr<scalar_t>();

    const auto num_threads = at::get_num_threads();
    const auto go_parallel =
        (num_edges * num_features) / num_threads >= at::internal::GRAIN_SIZE;
    const auto grain_size =
        (go_parallel) ? at::divup(num_edges, num_threads) : num_edges;
    at::parallel_for(0, num_edges, grain_size, [&](size_t beg, size_t end) {
      for (size_t i = beg; i < end; ++i) {
        const auto left_offset = left_index_base_ptr[i] * num_features;
        const auto right_offset = right_index_base_ptr[i] * num_features;
        const auto out_offset = i * num_features;
        const scalar_t* left_local_ptr = left_base_ptr + left_offset;
        const scalar_t* right_local_ptr = right_base_ptr + right_offset;
        scalar_t* out_local_ptr = out_base_ptr + out_offset;
        inner_loop(left_local_ptr, right_local_ptr, out_local_ptr,
                   num_features);
      }
    });
  });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::sampled_op"), TORCH_FN(sampled_op_kernel));
}

}  // namespace ops
}  // namespace pyg
