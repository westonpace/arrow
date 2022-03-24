#include "benchmark/benchmark.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "arrow/status.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/task_group.h"
#include "arrow/util/thread_pool.h"

namespace arrow {
namespace internal {

constexpr int kBatchSize = 1 << 13;
constexpr int kNumTasks = 32;

struct Workload {
  explicit Workload(int32_t size) : data_(size), size_(size), indices_(size) {
    std::default_random_engine gen(42);
    std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
    std::generate(data_.begin(), data_.end(), [&]() { return dist(gen); });

    std::iota(indices_.begin(), indices_.end(), 0);
    for (int offset = 0; offset < size; offset += kBatchSize) {
      std::shuffle(indices_.begin() + offset, indices_.begin() + offset + kBatchSize,
                   gen);
    }
  }

  void operator()(int, int);

 private:
  std::vector<uint64_t> data_;
  uint64_t size_;
  std::vector<uint64_t> indices_;
};

void Workload::operator()(int offset, int length) {
  uint64_t result = 0;
  uint64_t end = static_cast<uint64_t>(offset + length);
  for (uint64_t i = offset; i < end; ++i) {
    // result = (result << (data_[indices_[i]] % 64)) - data_[indices_[i]];
    result = (result << (data_[i] % 64)) - data_[i];
  }
  benchmark::DoNotOptimize(result);
}

struct Task {
  explicit Task(int32_t size) : workload_(size) {}

  Status operator()(int offset, int length) {
    workload_(offset, length);
    return Status::OK();
  }

 private:
  Workload workload_;
};

static void GroupedExecution(benchmark::State& state) {  // NOLINT non-const reference
  const auto workload_size = static_cast<int64_t>(state.range(0));

  Workload workload(workload_size);

  for (auto _ : state) {
    for (int i = 0; i < kNumTasks; i++) {
      workload(0, workload_size);
      workload(0, workload_size);
    }
  }
}

static void BatchedExecution(benchmark::State& state) {  // NOLINT non-const reference
  const auto workload_size = static_cast<int64_t>(state.range(0));
  Workload workload(workload_size);

  for (auto _ : state) {
    for (int offset = 0; offset < workload_size; offset += kBatchSize) {
      for (int i = 0; i < kNumTasks; i++) {
        workload(offset, kBatchSize);
      }
    }
  }
}

BENCHMARK(GroupedExecution)->RangeMultiplier(2)->Range(1 << 13, 1 << 23)->ThreadPerCpu();
BENCHMARK(BatchedExecution)->RangeMultiplier(2)->Range(1 << 13, 1 << 23)->ThreadPerCpu();
}  // namespace internal
}  // namespace arrow
