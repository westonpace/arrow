// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

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
#include "arrow/util/ws_thread_pool.h"

namespace arrow {
namespace internal {

struct Workload {
  explicit Workload(int32_t size) : size_(size), data_(kDataSize) {
    std::default_random_engine gen(42);
    std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
    std::generate(data_.begin(), data_.end(), [&]() { return dist(gen); });
  }

  void operator()();

 private:
  static constexpr int32_t kDataSize = 32;

  int32_t size_;
  std::vector<uint64_t> data_;
};

void Workload::operator()() {
  uint64_t result = 0;
  for (int32_t i = 0; i < size_ / kDataSize; ++i) {
    for (const auto v : data_) {
      result = (result << (v % 64)) - v;
    }
  }
  benchmark::DoNotOptimize(result);
}

struct Task {
  explicit Task(int32_t size) : workload_(size) {}

  Status operator()() {
    workload_();
    return Status::OK();
  }

 private:
  Workload workload_;
};

using ThreadPoolFactory = std::function<std::shared_ptr<ThreadPool>(int)>;

static const int32_t SIMPLE_THREAD_POOL = 1;
static const int32_t WORK_STEALING_THREAD_POOL = 2;
static const std::vector<int32_t> kThreadPoolImpls = {SIMPLE_THREAD_POOL,
                                                      WORK_STEALING_THREAD_POOL};

ThreadPoolFactory MakeSimple() {
  return [](int nthreads) { return *SimpleThreadPool::Make(nthreads); };
}

ThreadPoolFactory MakeWorkStealing() {
  return [](int nthreads) { return *WorkStealingThreadPool::Make(nthreads); };
}

static ThreadPoolFactory FactoryFromArg(int32_t arg) {
  switch (arg) {
    case SIMPLE_THREAD_POOL:
      return MakeSimple();
    case WORK_STEALING_THREAD_POOL:
      return MakeWorkStealing();
    default:
      assert(false);
  }
}

// Benchmark ThreadPool::Spawn
static void ThreadPoolSpawn(benchmark::State& state) {  // NOLINT non-const reference
  const auto thread_pool_factory = FactoryFromArg(state.range(0));
  const auto nthreads = static_cast<int>(state.range(1));
  const auto workload_size = static_cast<int32_t>(state.range(2));

  Workload workload(workload_size);

  // Spawn enough tasks to make the pool start up overhead negligible
  const int32_t nspawns = 200000000 / workload_size + 1;

  for (auto _ : state) {
    state.PauseTiming();
    std::shared_ptr<ThreadPool> pool;
    pool = thread_pool_factory(nthreads);
    state.ResumeTiming();

    for (int32_t i = 0; i < nspawns; ++i) {
      // Pass the task by reference to avoid copying it around
      ABORT_NOT_OK(pool->Spawn(std::ref(workload)));
    }

    // Wait for all tasks to finish
    ABORT_NOT_OK(pool->Shutdown(true /* wait */));
    state.PauseTiming();
    pool.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * nspawns);
}

// Benchmark ThreadPool::Spawn when the spawning happens withing the thread pool
static void ThreadPoolNestedSpawn(
    benchmark::State& state) {  // NOLINT non-const reference
  const auto thread_pool_factory = FactoryFromArg(state.range(0));
  const auto nthreads = static_cast<int>(state.range(1));
  const auto workload_size = static_cast<int32_t>(state.range(2));

  Workload workload(workload_size);

  // Spawn enough tasks to make the pool start up overhead negligible
  const int32_t nspawns = 200000000 / workload_size + 1;
  const int32_t spawns_per_thread = nspawns / nthreads;

  for (auto _ : state) {
    state.PauseTiming();
    std::shared_ptr<ThreadPool> pool;
    pool = thread_pool_factory(nthreads);
    state.ResumeTiming();

    for (int32_t i = 0; i < nthreads; ++i) {
      // Pass the task by reference to avoid copying it around
      ABORT_NOT_OK(pool->Spawn([&workload, &pool, spawns_per_thread] {
        for (int32_t j = 0; j < spawns_per_thread; j++) {
          pool->Spawn(std::ref(workload));
        }
      }));
    }

    // Wait for all tasks to finish
    ABORT_NOT_OK(pool->Shutdown(true /* wait */));
    state.PauseTiming();
    pool.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * spawns_per_thread * nthreads);
}

// Benchmark SerialExecutor::RunInSerialExecutor
static void RunInSerialExecutor(benchmark::State& state) {  // NOLINT non-const reference
  const auto workload_size = static_cast<int32_t>(state.range(0));

  Workload workload(workload_size);

  for (auto _ : state) {
    ABORT_NOT_OK(
        SerialExecutor::RunInSerialExecutor<Future<>>([&](internal::Executor* executor) {
          return DeferNotOk(executor->Submit(std::ref(workload)));
        }));
  }

  state.SetItemsProcessed(state.iterations());
}

// Benchmark ThreadPool::Submit
static void ThreadPoolSubmit(benchmark::State& state) {  // NOLINT non-const reference
  const auto thread_pool_factory = FactoryFromArg(state.range(0));
  const auto nthreads = static_cast<int>(state.range(1));
  const auto workload_size = static_cast<int32_t>(state.range(2));

  Workload workload(workload_size);

  const int32_t nspawns = 10000000 / workload_size + 1;

  for (auto _ : state) {
    state.PauseTiming();
    auto pool = thread_pool_factory(nthreads);
    std::atomic<int32_t> n_finished{0};
    state.ResumeTiming();

    for (int32_t i = 0; i < nspawns; ++i) {
      // Pass the task by reference to avoid copying it around
      (void)DeferNotOk(pool->Submit(std::ref(workload))).Then([&]() {
        n_finished.fetch_add(1);
      });
    }

    // Wait for all tasks to finish
    ABORT_NOT_OK(pool->Shutdown(true /* wait */));
    ASSERT_EQ(n_finished.load(), nspawns);
    state.PauseTiming();
    pool.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * nspawns);
}

// Benchmark serial TaskGroup
static void SerialTaskGroup(benchmark::State& state) {  // NOLINT non-const reference
  const auto workload_size = static_cast<int32_t>(state.range(0));

  Task task(workload_size);

  const int32_t nspawns = 10000000 / workload_size + 1;

  for (auto _ : state) {
    auto task_group = TaskGroup::MakeSerial();
    for (int32_t i = 0; i < nspawns; ++i) {
      // Pass the task by reference to avoid copying it around
      task_group->Append(std::ref(task));
    }
    ABORT_NOT_OK(task_group->Finish());
  }
  state.SetItemsProcessed(state.iterations() * nspawns);
}

// Benchmark threaded TaskGroup
static void ThreadedTaskGroup(benchmark::State& state) {  // NOLINT non-const reference
  const auto thread_pool_factory = FactoryFromArg(state.range(0));
  const auto nthreads = static_cast<int>(state.range(1));
  const auto workload_size = static_cast<int32_t>(state.range(2));

  std::shared_ptr<ThreadPool> pool;
  // FIXME: Add support for work-stealing
  pool = *SimpleThreadPool::Make(nthreads);

  Task task(workload_size);

  const int32_t nspawns = 10000000 / workload_size + 1;

  for (auto _ : state) {
    auto task_group = TaskGroup::MakeThreaded(pool.get());
    task_group->Append([&task, nspawns, task_group] {
      for (int32_t i = 0; i < nspawns; ++i) {
        // Pass the task by reference to avoid copying it around
        task_group->Append(std::ref(task));
      }
      return Status::OK();
    });
    ABORT_NOT_OK(task_group->Finish());
  }
  ABORT_NOT_OK(pool->Shutdown(true /* wait */));

  state.SetItemsProcessed(state.iterations() * nspawns);
}

static const std::vector<int32_t> kWorkloadSizes = {1000, 10000, 100000};

static void WorkloadCost_Customize(benchmark::internal::Benchmark* b) {
  for (const int32_t w : kWorkloadSizes) {
    b->Args({w});
  }
  b->ArgNames({"task_cost"});
  b->UseRealTime();
}

static void ThreadPoolSpawn_Customize(benchmark::internal::Benchmark* b) {
  for (const int32_t thread_pool_type : kThreadPoolImpls) {
    for (const int32_t w : kWorkloadSizes) {
      for (const int nthreads : {1, 2, 4, 8}) {
        b->Args({thread_pool_type, nthreads, w});
      }
    }
  }
  b->ArgNames({"impl", "threads", "task_cost"});
  b->UseRealTime();
}

#ifdef ARROW_WITH_BENCHMARKS_REFERENCE

// This benchmark simply provides a baseline indicating the raw cost of our workload
// depending on the workload size.  Number of items / second in this (serial)
// benchmark can be compared to the numbers obtained in ThreadPoolSpawn.
static void ReferenceWorkloadCost(benchmark::State& state) {
  const auto workload_size = static_cast<int32_t>(state.range(0));

  Workload workload(workload_size);
  for (auto _ : state) {
    workload();
  }

  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(ReferenceWorkloadCost)->Apply(WorkloadCost_Customize);

#endif

BENCHMARK(SerialTaskGroup)->Apply(WorkloadCost_Customize);
BENCHMARK(RunInSerialExecutor)->Apply(WorkloadCost_Customize);
BENCHMARK(ThreadPoolSpawn)->Apply(ThreadPoolSpawn_Customize);
BENCHMARK(ThreadPoolNestedSpawn)->Apply(ThreadPoolSpawn_Customize);
BENCHMARK(ThreadedTaskGroup)->Apply(ThreadPoolSpawn_Customize);
BENCHMARK(ThreadPoolSubmit)->Apply(ThreadPoolSpawn_Customize);

}  // namespace internal
}  // namespace arrow
