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

#pragma once

#ifndef _WIN32
#include <unistd.h>
#endif

#include <cstdint>
#include <memory>
#include <queue>
#include <type_traits>
#include <utility>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/cancel.h"
#include "arrow/util/executor.h"
#include "arrow/util/functional.h"
#include "arrow/util/future.h"
#include "arrow/util/macros.h"
#include "arrow/util/visibility.h"

#if defined(_MSC_VER)
// Disable harmless warning for decorated name length limit
#pragma warning(disable : 4503)
#endif

namespace arrow {

/// \brief Get the capacity of the global thread pool
///
/// Return the number of worker threads in the thread pool to which
/// Arrow dispatches various CPU-bound tasks.  This is an ideal number,
/// not necessarily the exact number of threads at a given point in time.
///
/// You can change this number using SetCpuThreadPoolCapacity().
ARROW_EXPORT int GetCpuThreadPoolCapacity();

/// \brief Set the capacity of the global thread pool
///
/// Set the number of worker threads int the thread pool to which
/// Arrow dispatches various CPU-bound tasks.
///
/// The current number is returned by GetCpuThreadPoolCapacity().
ARROW_EXPORT Status SetCpuThreadPoolCapacity(int threads);

namespace internal {

/// \brief An executor implementation that runs all tasks on a single thread using an
/// event loop.
///
/// Note: Any sort of nested parallelism will deadlock this executor.  Blocking waits are
/// fine but if one task needs to wait for another task it must be expressed as an
/// asynchronous continuation.
class ARROW_EXPORT SerialExecutor : public Executor {
 public:
  template <typename T = ::arrow::internal::Empty>
  using TopLevelTask = internal::FnOnce<Future<T>(Executor*)>;

  ~SerialExecutor();

  int GetCapacity() override { return 1; };
  bool HasIdleCapacity() override { return false; }
  Status SpawnReal(TaskHints hints, FnOnce<void()> task, StopToken,
                   StopCallback&&) override;

  /// \brief Runs the TopLevelTask and any scheduled tasks
  ///
  /// The TopLevelTask (or one of the tasks it schedules) must either return an invalid
  /// status or call the finish signal. Failure to do this will result in a deadlock.  For
  /// this reason it is preferable (if possible) to use the helper methods (below)
  /// RunSynchronously/RunSerially which delegates the responsiblity onto a Future
  /// producer's existing responsibility to always mark a future finished (which can
  /// someday be aided by ARROW-12207).
  template <typename T = internal::Empty, typename FT = Future<T>,
            typename FTSync = typename FT::SyncType>
  static FTSync RunInSerialExecutor(TopLevelTask<T> initial_task) {
    return SerialExecutor().Run<T>(std::move(initial_task)).to_sync();
  }

 private:
  SerialExecutor();

  // State uses mutex
  struct State;
  std::shared_ptr<State> state_;

  template <typename T, typename FTSync = typename Future<T>::SyncType>
  Future<T> Run(TopLevelTask<T> initial_task) {
    auto final_fut = std::move(initial_task)(this);
    if (final_fut.is_finished()) {
      return final_fut;
    }
    final_fut.AddCallback([this](const FTSync&) { MarkFinished(); });
    RunLoop();
    return final_fut;
  }
  void RunLoop();
  void MarkFinished();
};

/// An Executor implementation spawning tasks in FIFO manner on a fixed-size
/// pool of worker threads.
///
/// Note: Any sort of nested parallelism will deadlock this executor.  Blocking waits are
/// fine but if one task needs to wait for another task it must be expressed as an
/// asynchronous continuation.
class ARROW_EXPORT ThreadPool : public Executor {
 public:
  // Construct a thread pool with the given number of worker threads
  static Result<std::shared_ptr<ThreadPool>> Make(int threads);

  // Like Make(), but takes care that the returned ThreadPool is compatible
  // with destruction late at process exit.
  static Result<std::shared_ptr<ThreadPool>> MakeEternal(int threads);

  // Destroy thread pool; the pool will first be shut down
  ~ThreadPool() override;

  // Return the desired number of worker threads.
  // The actual number of workers may lag a bit before being adjusted to
  // match this value.
  int GetCapacity() override;

  // Return true if there are no tasks in the queue and at least one thread
  // is idle.
  bool HasIdleCapacity() override;

  // Return the number of tasks either running or in the queue.
  int GetNumTasks();

  // Dynamically change the number of worker threads.
  //
  // This function always returns immediately.
  // If fewer threads are running than this number, new threads are spawned
  // on-demand when needed for task execution.
  // If more threads are running than this number, excess threads are reaped
  // as soon as possible.
  Status SetCapacity(int threads);

  // Heuristic for the default capacity of a thread pool for CPU-bound tasks.
  // This is exposed as a static method to help with testing.
  static int DefaultCapacity();

  // Shutdown the pool.  Once the pool starts shutting down, new tasks
  // cannot be submitted anymore.
  // If "wait" is true, shutdown waits for all pending tasks to be finished.
  // If "wait" is false, workers are stopped as soon as currently executing
  // tasks are finished.
  Status Shutdown(bool wait = true);

  struct State;

 protected:
  FRIEND_TEST(TestThreadPool, SetCapacity);
  FRIEND_TEST(TestGlobalThreadPool, Capacity);
  friend ARROW_EXPORT ThreadPool* GetCpuThreadPool();

  ThreadPool();

  Status SpawnReal(TaskHints hints, FnOnce<void()> task, StopToken,
                   StopCallback&&) override;

  // Collect finished worker threads, making sure the OS threads have exited
  void CollectFinishedWorkersUnlocked();
  // Launch a given number of additional workers
  void LaunchWorkersUnlocked(int threads);
  // Get the current actual capacity
  int GetActualCapacity();
  // Reinitialize the thread pool if the pid changed
  void ProtectAgainstFork();

  static std::shared_ptr<ThreadPool> MakeCpuThreadPool();

  std::shared_ptr<State> sp_state_;
  State* state_;
  bool shutdown_on_destroy_;
#ifndef _WIN32
  pid_t pid_;
#endif
};

// Return the process-global thread pool for CPU-bound tasks.
ARROW_EXPORT ThreadPool* GetCpuThreadPool();

/// \brief Potentially run an async operation serially (if use_threads is false)
/// \see RunSerially
///
/// If `use_threads` is true, the global CPU executor is used.
/// If `use_threads` is false, a temporary SerialExecutor is used.
/// `get_future` is called (from this thread) with the chosen executor and must
/// return a future that will eventually finish. This function returns once the
/// future has finished.
template <typename Fut, typename ValueType = typename Fut::ValueType>
typename Fut::SyncType RunSynchronously(FnOnce<Fut(Executor*)> get_future,
                                        bool use_threads) {
  if (use_threads) {
    return std::move(get_future)(GetCpuThreadPool()).to_sync();
  } else {
    return SerialExecutor::RunInSerialExecutor<ValueType>(std::move(get_future));
  }
}

}  // namespace internal
}  // namespace arrow
