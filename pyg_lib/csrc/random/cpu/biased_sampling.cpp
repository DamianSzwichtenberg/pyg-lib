#include "pyg_lib/csrc/random/cpu/biased_sampling.h"

#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

namespace pyg {
namespace random {

c10::optional<at::Tensor> biased_to_cdf(const at::Tensor& rowptr,
                                        at::Tensor& bias,
                                        bool inplace) {
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(bias.is_cpu(), "'bias' must be a CPU tensor");

  auto cdf = at::empty_like(bias);
  // TODO: Also dispatch index type
  size_t rowptr_size = rowptr.size(0);
  int64_t* rowptr_data = rowptr.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES(bias.scalar_type(), "biased_to_cdf", [&] {
    const auto bias_data = bias.data_ptr<scalar_t>();
    scalar_t* cdf_data = nullptr;
    if (inplace) {
      cdf_data = bias.data_ptr<scalar_t>();
    } else {
      cdf_data = cdf.data_ptr<scalar_t>();
    }
    biased_to_cdf_helper(rowptr_data, rowptr_size, bias_data, cdf_data);
  });

  return cdf;
}

// The implementation of coverting to CDF representation for biased sampling.
template <typename scalar_t>
void biased_to_cdf_helper(int64_t* rowptr_data,
                          size_t rowptr_size,
                          const scalar_t* bias,
                          scalar_t* cdf) {
  using Vec = at::vec::Vectorized<scalar_t>;

  at::parallel_for(
      0, rowptr_size - 1, at::internal::GRAIN_SIZE / rowptr_size,
      [&](int64_t _s, int64_t _e) {
        // CDF conversion for each row
        for (int64_t i = _s; i < _e; i++) {
          const scalar_t* in_beg = bias + rowptr_data[i];
          scalar_t* out_beg = cdf + rowptr_data[i];
          int64_t len = rowptr_data[i + 1] - rowptr_data[i];
          std::vector<scalar_t> shifted_prefix_sum(len,
                                                   static_cast<scalar_t>(0));

          for (int64_t j = 1; j < len; ++j) {
            shifted_prefix_sum[j] = shifted_prefix_sum[j - 1] + in_beg[j - 1];
          }
          scalar_t sum = shifted_prefix_sum[len - 1] + in_beg[len - 1];
          scalar_t* pref_beg = shifted_prefix_sum.data();

          int64_t d = 0;
          for (; d < len - (len % Vec::size()); d += Vec::size()) {
            Vec data_vec = Vec::loadu(pref_beg + d);
            Vec data_vec2 = Vec(sum);
            Vec output_vec = data_vec / data_vec2;
            output_vec.store(out_beg + d);
          }
          if (len - d > 0) {
            Vec data_vec = Vec::loadu(pref_beg + d, len - d);
            Vec data_vec2 = Vec(sum);
            Vec output_vec = data_vec / data_vec2;
            output_vec.store(out_beg + d, len - d);
          }
        }
      });
}

std::pair<at::Tensor, at::Tensor> biased_to_alias(at::Tensor rowptr,
                                                  at::Tensor bias) {
  TORCH_CHECK(rowptr.is_cpu(), "'rowptr' must be a CPU tensor");
  TORCH_CHECK(bias.is_cpu(), "'bias' must be a CPU tensor");

  at::Tensor alias = at::empty_like(bias, rowptr.options());
  at::Tensor out_bias = bias.clone();

  size_t rowptr_size = rowptr.size(0);
  int64_t* rowptr_data = rowptr.data_ptr<int64_t>();
  int64_t* alias_data = alias.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES(bias.scalar_type(), "biased_to_cdf_inplace", [&] {
    scalar_t* bias_data = bias.data_ptr<scalar_t>();
    scalar_t* out_bias_data = out_bias.data_ptr<scalar_t>();
    biased_to_alias_helper(rowptr_data, rowptr_size, bias_data, out_bias_data,
                           alias_data);
  });

  return {out_bias, alias};
}

template <typename scalar_t>
void biased_to_alias_helper(int64_t* rowptr_data,
                            size_t rowptr_size,
                            const scalar_t* bias,
                            scalar_t* out_bias,
                            int64_t* alias) {
  scalar_t eps = 1e-6;
  at::parallel_for(
      0, rowptr_size - 1, at::internal::GRAIN_SIZE / rowptr_size,
      [&](size_t _s, size_t _e) {
        // Calculate the average bias
        for (size_t i = _s; i < _e; i++) {
          const scalar_t* beg = bias + rowptr_data[i];
          size_t len = rowptr_data[i + 1] - rowptr_data[i];
          scalar_t* out_beg = out_bias + rowptr_data[i];
          int64_t* alias_beg = alias + rowptr_data[i];
          scalar_t avg = 0;
          size_t j_;
#ifdef _OPENMP
#pragma omp simd reduction(+ : avg) linear(j_ : 1)
#endif
          for (j_ = 0; j_ < len; j_++) {
            avg += beg[j_];
          }
          avg /= len;

          // The sets for index with a bias lower or higher than average
          std::vector<size_t> high, low;

          low.reserve(len / 2 + 1);
          high.reserve(len / 2 + 1);
#ifdef _OPENMP
#pragma omp simd
#endif
          for (size_t j = 0; j < len; j++) {
            scalar_t b = beg[j];
            // Allow some floating point error
            if (b > avg + eps) {
              high.push_back(j);
            } else if (b < avg - eps) {
              low.push_back(j);
            } else {  // if close to avg, make it a stable entry
              out_beg[j] = 1;
              alias_beg[j] = j;
            }
          }

          // Keep merging two elements, one from the lower bias set and the
          // other from the higher bias set.
          while (!low.empty()) {
            auto low_idx = low.back();

            // An index with bias lower than average means another higher one.
            TORCH_CHECK(
                !high.empty(),
                "every bias lower than avg should have a higher counterpart");
            auto high_idx = high.back();
            low.pop_back();
            high.pop_back();

            // Handle the lower one:
            auto low_bias = out_beg[low_idx];
            out_beg[low_idx] = low_bias / avg;
            alias_beg[low_idx] = high_idx;
            out_beg[high_idx] -= avg - low_bias;

            // Handle the higher one:
            scalar_t high_bias_left = out_beg[high_idx];

            // Dispatch the remaining bias to the corresponding set.
            if (high_bias_left > avg + eps) {
              high.push_back(high_idx);
            } else if (high_bias_left < avg - eps) {
              low.push_back(high_idx);
            } else {
              out_beg[high_idx] = 1;
              alias_beg[high_idx] = high_idx;
            }
          }
        }
      });
}

}  // namespace random

}  // namespace pyg
