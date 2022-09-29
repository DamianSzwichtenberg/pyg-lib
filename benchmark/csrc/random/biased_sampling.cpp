#include <algorithm>
#include <random>
#include <type_traits>
#include <vector>

#include <ATen/ATen.h>
#include <benchmark/benchmark.h>

#include "pyg_lib/csrc/random/cpu/biased_sampling.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"

template <typename scalar_t>
class BenchmarkBiasedSampling : public benchmark::Fixture {
  static_assert(std::is_floating_point<scalar_t>::value,
                "Floating type required for scalar_t");

 protected:
  virtual void PerformTest(const at::Tensor& rowptr, at::Tensor& bias) = 0;

  void SetUp(const benchmark::State& state) override {
    num_nodes_ = state.range(0);
    group_size_ = state.range(1);

    rowptr_.resize(num_nodes_ + 1);
    std::iota(rowptr_.begin(), rowptr_.end(), 0);
    std::for_each(rowptr_.begin(), rowptr_.end(),
                  [this](int64_t& v) { v *= this->group_size_; });

    bias_.resize(num_nodes_ * group_size_);
    std::mt19937 gen;
    for (int64_t i = 0; i < num_nodes_; ++i) {
      std::generate(bias_.begin() + (i * group_size_),
                    bias_.begin() + ((i + 1) * group_size_),
                    [v = 0.0]() mutable { return v += 0.125; });
      std::shuffle(bias_.begin() + (i * group_size_),
                   bias_.begin() + ((i + 1) * group_size_), gen);
    }
  }

  void TearDown(const benchmark::State& state) override {
    rowptr_.clear();
    bias_.clear();
  }

  void Loop(benchmark::State& state) {
    at::Tensor rowptr = pyg::utils::from_vector<int64_t>(rowptr_);
    at::Tensor bias = pyg::utils::from_vector<scalar_t>(bias_);
    for (auto _ : state) {
      PerformTest(rowptr, bias);
    }
  }

 private:
  int64_t num_nodes_;
  int64_t group_size_;
  std::vector<int64_t> rowptr_;
  std::vector<scalar_t> bias_;
};

template <typename scalar_t>
class BiasedToCDF : public BenchmarkBiasedSampling<scalar_t> {
 protected:
  void PerformTest(const at::Tensor& rowptr, at::Tensor& bias) override {
    const auto result = pyg::random::biased_to_cdf(rowptr, bias, false);
    benchmark::DoNotOptimize(result);
  }
};

template <typename scalar_t>
class BiasedToAlias : public BenchmarkBiasedSampling<scalar_t> {
 protected:
  void PerformTest(const at::Tensor& rowptr, at::Tensor& bias) override {
    const auto result = pyg::random::biased_to_alias(rowptr, bias);
    benchmark::DoNotOptimize(result);
  }
};

BENCHMARK_TEMPLATE_DEFINE_F(BiasedToCDF, BasicBenchmark, float)
(benchmark::State& state) {
  Loop(state);
}
BENCHMARK_REGISTER_F(BiasedToCDF, BasicBenchmark)
    // ->ArgsProduct({{benchmark::CreateRange(16, 2048, 2)},
    ->ArgsProduct({{8}, {benchmark::CreateRange(2, 64, 2)}})
    ->ArgNames({"num_nodes", "group_size"});
// for now, test only single-threaded cases
// some cases may use multiple threads under the hood
// ->MeasureProcessCPUTime();

BENCHMARK_TEMPLATE_DEFINE_F(BiasedToAlias, BasicBenchmark, float)
(benchmark::State& state) {
  Loop(state);
}
BENCHMARK_REGISTER_F(BiasedToAlias, BasicBenchmark)
    // ->ArgsProduct({{benchmark::CreateRange(16, 2048, 2)},
    ->ArgsProduct({{8}, {benchmark::CreateRange(2, 64, 2)}})
    ->ArgNames({"num_nodes", "group_size"});
// for now, test only single-threaded cases
// some cases may use multiple threads under the hood
// ->MeasureProcessCPUTime();