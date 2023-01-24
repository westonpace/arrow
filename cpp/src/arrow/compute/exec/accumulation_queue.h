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

#include <cstdint>
#include <functional>
#include <vector>

#include "arrow/compute/exec.h"
#include "arrow/result.h"
#include "arrow/util/functional.h"

namespace arrow {

using internal::FnOnce;

namespace util {
using arrow::compute::ExecBatch;

/// \brief A container that accumulates batches until they are ready to
///        be processed.
class AccumulationQueue {
 public:
  AccumulationQueue() : row_count_(0) {}
  ~AccumulationQueue() = default;

  // We should never be copying ExecBatch around
  AccumulationQueue(const AccumulationQueue&) = delete;
  AccumulationQueue& operator=(const AccumulationQueue&) = delete;

  AccumulationQueue(AccumulationQueue&& that);
  AccumulationQueue& operator=(AccumulationQueue&& that);

  void Concatenate(AccumulationQueue&& that);
  void InsertBatch(ExecBatch batch);
  int64_t row_count() { return row_count_; }
  size_t batch_count() { return batches_.size(); }
  bool empty() const { return batches_.empty(); }
  void Clear();
  ExecBatch& operator[](size_t i);

 private:
  int64_t row_count_;
  std::vector<ExecBatch> batches_;
};

/// An queue that sequences incoming batches
///
/// This can be used when a node needs to do some kind of ordered processing on
/// the stream.
///
/// Batches can be inserted in any order.  The process_callback will be called on
/// the batches, in order, without reentrant calls. For this reason the callback
/// should generally be fairly quick.
///
/// For example, in a top-n node, the process callback should determine how many
/// rows need to be delivered for the given batch, and then return a task to actually
/// deliver those rows.
class SequencingQueue {
 public:
  using Task = FnOnce<Status()>;
  using ProcessCallback = std::function<Result<Task>(ExecBatch)>;
  using ScheduleCallback = std::function<void(Task)>;

  virtual ~SequencingQueue() = default;

  /// @brief Insert a batch into the queue
  ///
  /// This will insert the batch into the queue.  If this batch was the next batch
  /// to deliver then this will trigger 1+ calls to the process callback.  Each of
  /// those calls will generate a task which will then be executed.  Finally, once
  /// all generated tasks have been executed, this function will return.
  virtual Status InsertBatch(ExecBatch batch) = 0;

  static std::unique_ptr<SequencingQueue> Make(ProcessCallback process_callback,
                                               ScheduleCallback schedule_callback);
};

}  // namespace util
}  // namespace arrow
