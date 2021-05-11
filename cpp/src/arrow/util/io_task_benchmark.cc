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

#include "arrow/io/file.h"
#include "arrow/status.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/io_util.h"
#include "arrow/util/iterator.h"
#include "arrow/util/task_group.h"
#include "arrow/util/thread_pool.h"

namespace arrow {
namespace internal {

constexpr int64_t GIGABYTE = 100 * 1024 * 1024;
constexpr int64_t TESTDATA_SIZE = 2 * GIGABYTE;
constexpr auto kSeed = 0x42;

Status WriteData(const std::string& path) {
  ARROW_ASSIGN_OR_RAISE(auto outstream, io::FileOutputStream::Open(path));
  std::shared_ptr<Buffer> data = *AllocateBuffer(TESTDATA_SIZE);
  random_bytes(TESTDATA_SIZE, kSeed, data->mutable_data());
  RETURN_NOT_OK(outstream->Write(data));
  RETURN_NOT_OK(outstream->Close());
  return arrow::internal::FileSync(outstream->file_descriptor());
}

Result<AsyncGenerator<std::shared_ptr<Buffer>>> MakeSource(
    const std::string& path, int64_t block_size, ShouldSchedule should_schedule) {
  ARROW_ASSIGN_OR_RAISE(auto instream, io::ReadableFile::Open(path));
  ARROW_ASSIGN_OR_RAISE(auto it,
                        io::MakeInputStreamIterator(std::move(instream), block_size));
  ARROW_ASSIGN_OR_RAISE(
      auto gen,
      MakeBackgroundGenerator(std::move(it), io::default_io_context().executor()));
  gen = MakeSerialReadaheadGenerator(std::move(gen),
                                     internal::GetCpuThreadPool()->GetCapacity() * 2);
  auto transferred = MakeTransferredGenerator(
      std::move(gen), internal::GetCpuThreadPool(), should_schedule);
  return transferred;
}

std::function<Result<util::optional<int64_t>>(const std::shared_ptr<Buffer>&)>
MakeWorkload() {
  return [](const std::shared_ptr<Buffer>& buf) {
    int64_t local_sum = 0;
    auto it = buf->data();
    auto end = it + buf->size();
    for (int i = 0; i < 2; i++) {
      for (; it != end; it++) {
        local_sum = (local_sum << (*it % 64)) - *it - i;
      }
      __asm__ __volatile__("");
    }
    return local_sum;
  };
}

static void IoTaskBasicBench(benchmark::State& state) {  // NOLINT non-const reference
  // const auto nthreads = static_cast<int>(state.range(0));
  // constexpr int64_t BLOCK_SIZE = 1024 * 1024;
  constexpr int64_t BLOCK_SIZE = 8 * 1024 * 1024;
  const auto should_schedule = static_cast<ShouldSchedule>(state.range(0));

  ASSERT_OK_AND_ASSIGN(auto tempdir,
                       arrow::internal::TemporaryDir::Make("io-task-benchmark-"));
  ASSERT_OK_AND_ASSIGN(auto path, tempdir->path().Join("data"));
  ASSERT_OK(WriteData(path.ToString()));

  int64_t accumulator = 0;
  int64_t num_calls = 0;
  std::function<Status(util::optional<int64_t>)> visitor =
      [&accumulator, &num_calls](util::optional<int64_t> value) {
        accumulator += *value;
        num_calls++;
        return Status::OK();
      };

  auto workload = MakeWorkload();

  int old_tasks_scheduled = internal::GetCpuThreadPool()->GetNumTasksScheduled();
  int old_io_tasks_scheduled =
      static_cast<internal::ThreadPool*>(io::default_io_context().executor())
          ->GetNumTasksScheduled();

  for (auto _ : state) {
    state.PauseTiming();
    ASSERT_OK_AND_ASSIGN(auto gen,
                         MakeSource(path.ToString(), BLOCK_SIZE, should_schedule));
    auto mapped = MakeMappedGenerator<std::shared_ptr<Buffer>, util::optional<int64_t>>(
        std::move(gen), workload);
    auto readahead = MakeReadaheadGenerator(std::move(mapped),
                                            internal::GetCpuThreadPool()->GetCapacity());
    state.ResumeTiming();
    VisitAsyncGenerator(std::move(readahead), visitor).Wait();
  }

  int cpu_tasks_scheduled =
      internal::GetCpuThreadPool()->GetNumTasksScheduled() - old_tasks_scheduled;
  int io_tasks_scheduled =
      static_cast<internal::ThreadPool*>(io::default_io_context().executor())
          ->GetNumTasksScheduled() -
      old_io_tasks_scheduled;
  state.counters["# CPU Tasks"] = cpu_tasks_scheduled / state.iterations();
  state.counters["# I/O Tasks"] = io_tasks_scheduled / state.iterations();
  state.counters["# Calls"] = num_calls / state.iterations();
}  // namespace internal

BENCHMARK(IoTaskBasicBench)
    //    ->Arg(ShouldSchedule::IF_UNFINISHED)
    ->Arg(ShouldSchedule::ALWAYS)
    //    ->Arg(ShouldSchedule::IF_IDLE)
    ->UseRealTime();

}  // namespace internal
}  // namespace arrow
