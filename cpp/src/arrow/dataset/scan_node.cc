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

#include "arrow/dataset/scanner.h"

#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "arrow/compute/exec/exec_plan.h"
#include "arrow/compute/exec/expression.h"
#include "arrow/dataset/scanner.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/make_unique.h"
#include "arrow/util/tracing_internal.h"
#include "arrow/util/unreachable.h"

namespace cp = arrow::compute;

namespace arrow {

using internal::checked_cast;

namespace dataset {

namespace {

Result<std::shared_ptr<Schema>> OutputSchemaFromOptions(const ScanV2Options& options) {
  return FieldPath::GetAll(*options.dataset->schema(), options.columns);
}

class BatchOutputStrategy {
 public:
  BatchOutputStrategy(compute::ExecNode* self, compute::ExecNode* output)
      : self_(self), output_(output) {}
  virtual ~BatchOutputStrategy() = default;
  virtual void DeliverBatch(compute::ExecBatch batch) = 0;
  virtual void InputFinished() {
    // std::cout << "BatchOutputFinished: " + std::to_string(batch_count_.load()) + "\n";
    output_->InputFinished(self_, batch_count_.load());
  }

 protected:
  void SliceAndDoDeliverBatch(compute::ExecBatch batch) {
    // FIXME - Add slicing
    batch_count_++;
    output_->InputReceived(self_, std::move(batch));
  }

  std::atomic<int32_t> batch_count_{0};
  compute::ExecNode* self_;
  compute::ExecNode* output_;
};

class UnorderedBatchOutputStrategy : public BatchOutputStrategy {
 public:
  UnorderedBatchOutputStrategy(compute::ExecNode* self, compute::ExecNode* output)
      : BatchOutputStrategy(self, output){};
  ~UnorderedBatchOutputStrategy() = default;
  void DeliverBatch(compute::ExecBatch batch) override {
    return SliceAndDoDeliverBatch(std::move(batch));
  }
};

// Unlike other nodes the scanner has to synchronize a number of resources
// that are external to the plan itself.  These things may start tasks that
// are invisible to the exec plan.
class ScannerSynchronization {
 public:
  ScannerSynchronization(int fragment_readahead)
      : fragment_readahead_(fragment_readahead) {}

  bool MarkListingCompleted() {
    std::lock_guard<std::mutex> lg(mutex_);
    finished_listing_ = true;
    return fragments_active_ == 0;
  }

  void MarkListingFailed(Status err) {
    std::lock_guard<std::mutex> lg(mutex_);
    finished_listing_ = true;
    cancelled_ = true;
    error_ = std::move(err);
  }

  void Cancel() {
    std::lock_guard<std::mutex> lg(mutex_);
    if (!cancelled_) {
      cancelled_ = true;
      error_ = Status::Cancelled("Plan cancelled before scan completed");
    }
  }

  void Pause() {
    std::lock_guard<std::mutex> lg(mutex_);
    paused_ = true;
  }

  void ReportFailedFragment(Status st) {
    std::lock_guard<std::mutex> lg(mutex_);
    if (!cancelled_) {
      cancelled_ = true;
      error_ = std::move(st);
      fragments_active_--;
    }
  }

  struct NextFragment {
    std::shared_ptr<Fragment> fragment;
    bool should_mark_input_finished;
  };

  NextFragment FinishFragmentAndGetNext() {
    std::lock_guard<std::mutex> lg(mutex_);
    if (cancelled_ || paused_ || fragments_to_scan_.empty()) {
      fragments_active_--;
      // std::cout << "Not continuing fragment inline fragments_active=" +
      //                  std::to_string(fragments_active_) + "\n";
      return {nullptr, /*should_mark_input_finished=*/finished_listing_};
    }
    // std::cout << "Continuing fragment inline " + std::to_string(fragments_active_) +
    // "\n";
    std::shared_ptr<Fragment> next = std::move(fragments_to_scan_.front());
    fragments_to_scan_.pop();
    return {next, /*should_mark_input_finished=*/false};
  }

  bool QueueFragment(std::shared_ptr<Fragment> fragment) {
    std::lock_guard<std::mutex> lg(mutex_);
    if (cancelled_) {
      return false;
    }
    if (!paused_ && fragments_active_ < fragment_readahead_) {
      // If we have space start scanning the fragment immediately
      fragments_active_++;
      // std::cout << "Start scanning immediately: " + std::to_string(fragments_active_) +
      //                  "\n";
      return true;
    }
    fragments_to_scan_.push(std::move(fragment));
    return false;
  }

 private:
  int fragment_readahead_;
  std::mutex mutex_;
  std::queue<std::shared_ptr<Fragment>> fragments_to_scan_;
  Status error_;
  bool cancelled_ = false;
  bool finished_listing_ = false;
  int fragments_active_ = 0;
  bool paused_ = false;
};

// In the future we should support async scanning of fragments.  The
// Dataset class doesn't support this yet but we pretend it does here to
// ease future adoption of the feature.
AsyncGenerator<std::shared_ptr<Fragment>> GetFragments(Dataset* dataset,
                                                       cp::Expression predicate) {
  // In the future the dataset should be responsible for figuring out
  // the I/O context.  This will allow different I/O contexts to be used
  // when scanning different datasets.  For example, if we are scanning a
  // union of a remote dataset and a local dataset.
  const auto& io_context = io::default_io_context();
  auto io_executor = io_context.executor();
  Future<std::shared_ptr<FragmentIterator>> fragments_it_fut =
      DeferNotOk(io_executor->Submit(
          [dataset, predicate]() -> Result<std::shared_ptr<FragmentIterator>> {
            // std::cout << "dataset->GetFragments\n";
            ARROW_ASSIGN_OR_RAISE(FragmentIterator fragments_iter,
                                  dataset->GetFragments(predicate));
            return std::make_shared<FragmentIterator>(std::move(fragments_iter));
          }));
  Future<AsyncGenerator<std::shared_ptr<Fragment>>> fragments_gen_fut =
      fragments_it_fut.Then([](const std::shared_ptr<FragmentIterator>& fragments_it)
                                -> Result<AsyncGenerator<std::shared_ptr<Fragment>>> {
        ARROW_ASSIGN_OR_RAISE(std::vector<std::shared_ptr<Fragment>> fragments,
                              fragments_it->ToVector());
        return MakeVectorGenerator(std::move(fragments));
      });
  return MakeFromFuture(std::move(fragments_gen_fut));
}

/// \brief A node that scans a dataset
///
/// The scan node has three groups of io-tasks and one task.
///
/// The first io-task (listing) fetches the fragments from the dataset.  This may be a
/// simple iteration of paths or, if the dataset is described with wildcards, this may
/// involve I/O for listing and walking directory paths.  There is one listing io-task per
/// dataset.
///
/// Ths next step is to fetch the metadata for the fragment.  For some formats (e.g. CSV)
/// this may be quite simple (get the size of the file).  For other formats (e.g. parquet)
/// this is more involved and requires reading data.  There is one metadata io-task per
/// fragment.  The metadata io-task creates an AsyncGenerator<RecordBatch> from the
/// fragment.
///
/// Once the metadata io-task is done we can issue read io-tasks.  Each read io-task
/// requests a single batch of data from the disk by pulling the next Future from the
/// generator.
///
/// Finally, when the future is fulfilled, we issue a pipeline task to drive the batch
/// through the pipeline.
///
/// Most of these tasks are io-tasks.  They take very few CPU resources and they run on
/// the I/O thread pool.  These io-tasks are invisible to the exec plan and so we need to
/// do some custom scheduling.  We limit how many fragments we read from at any one time.
/// This is referred to as "fragment readahead".
///
/// Within a fragment there is usually also some amount of "row readahead".  This row
/// readahead is handled by the fragment (and not the scanner) because the exact details
/// of how it is performed depend on the underlying format.
///
/// When a scan node is aborted (StopProducing) we send a cancel signal to any active
/// fragments.  On destruction we continue consuming the fragments until they complete
/// (which should be fairly quick since we cancelled the fragment).  This ensures the
/// I/O work is completely finished before the node is destroyed.
class ScanNode : public cp::ExecNode {
 public:
  ScanNode(cp::ExecPlan* plan, ScanV2Options options,
           std::shared_ptr<Schema> output_schema)
      : cp::ExecNode(plan, {}, {}, std::move(output_schema),
                     /*num_outputs=*/1),
        options_(options),
        synchronization_(options_.fragment_readahead) {}

  static Result<ScanV2Options> NormalizeAndValidate(const ScanV2Options& options,
                                                    compute::ExecContext* ctx) {
    ScanV2Options normalized(options);
    if (!normalized.dataset) {
      return Status::Invalid("Scan options must include a dataset");
    }

    if (normalized.filter.IsBound()) {
      // There is no easy way to make sure a filter was bound agaisnt the same
      // function registry as the one in ctx so we just require it to be unbound
      // FIXME - Do we care if it was bound to a different function registry?
      return Status::Invalid("Scan filter must be unbound");
    } else {
      ARROW_ASSIGN_OR_RAISE(normalized.filter,
                            normalized.filter.Bind(*options.dataset->schema(), ctx));
    }

    return std::move(normalized);
  }

  static Result<cp::ExecNode*> Make(cp::ExecPlan* plan, std::vector<cp::ExecNode*> inputs,
                                    const cp::ExecNodeOptions& options) {
    RETURN_NOT_OK(ValidateExecNodeInputs(plan, inputs, 0, "ScanNode"));
    const auto& scan_options = checked_cast<const ScanV2Options&>(options);
    ARROW_ASSIGN_OR_RAISE(ScanV2Options normalized_options,
                          NormalizeAndValidate(scan_options, plan->exec_context()));
    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Schema> output_schema,
                          OutputSchemaFromOptions(normalized_options));
    return plan->EmplaceNode<ScanNode>(plan, std::move(normalized_options),
                                       std::move(output_schema));
  }

  const char* kind_name() const override { return "ScanNode"; }

  [[noreturn]] static void NoInputs() {
    Unreachable("no inputs; this should never be called");
  }
  [[noreturn]] void InputReceived(cp::ExecNode*, cp::ExecBatch) override { NoInputs(); }
  [[noreturn]] void ErrorReceived(cp::ExecNode*, Status) override { NoInputs(); }
  [[noreturn]] void InputFinished(cp::ExecNode*, int) override { NoInputs(); }

  Status Init() override {
    batch_output_ =
        ::arrow::internal::make_unique<UnorderedBatchOutputStrategy>(this, outputs_[0]);
    return Status::OK();
  }

  Status StartProducing() override {
    START_COMPUTE_SPAN(span_, std::string(kind_name()) + ":" + label(),
                       {{"node.kind", kind_name()},
                        {"node.label", label()},
                        {"node.output_schema", output_schema()->ToString()},
                        {"node.detail", ToString()}});
    END_SPAN_ON_FUTURE_COMPLETION(span_, finished_);
    // std::cout << "StartProducing\n";
    ARROW_ASSIGN_OR_RAISE(Future<> list_task, plan_->BeginExternalTask());
    if (!list_task.is_valid()) {
      // Cancelled before we even started
      return Status::OK();
    }
    // std::cout << "LIST TASK: BEGIN\n";

    AsyncGenerator<std::shared_ptr<Fragment>> frag_gen =
        GetFragments(options_.dataset.get(), options_.filter);
    Future<> visit_task = VisitAsyncGenerator(
        std::move(frag_gen), [this](const std::shared_ptr<Fragment>& fragment) {
          // std::cout << "Queuing fragment: " + fragment->ToString() + "\n";
          QueueOrStartFragment(fragment);
          return Status::OK();
        });
    visit_task
        .Then(
            [this]() {
              // std::cout << "Marking listing completed\n";
              if (synchronization_.MarkListingCompleted()) {
                // It's possible, but probably unlikely, that all fragment scans have
                // already finished and so we need to report the total batch count here in
                // this case
                batch_output_->InputFinished();
                finished_.MarkFinished();
              }
            },
            [this](const Status& err) {
              synchronization_.MarkListingFailed(err);
              outputs_[0]->ErrorReceived(this, err);
            })
        .AddCallback([list_task](const Status& st) mutable {
          // std::cout << "LIST TASK: END\n";
          list_task.MarkFinished();
        });

    return Status::OK();
  }

  void PauseProducing(ExecNode* output, int32_t counter) override {
    // FIXME(TODO)
    // Need to ressurect AsyncToggle and then all fragment scanners
    // should share the same toggle
  }

  void ResumeProducing(ExecNode* output, int32_t counter) override {
    // FIXME(TODO)
  }

  void StopProducing(ExecNode* output) override {
    DCHECK_EQ(output, outputs_[0]);
    StopProducing();
  }

  void StopProducing() override { synchronization_.Cancel(); }

 private:
  struct FragmentScanTask;
  using ScanTaskPtr = std::list<FragmentScanTask>::iterator;
  struct FragmentScanTask {
    std::unique_ptr<FragmentEvolutionStrategy> fragment_evolution;
    std::shared_ptr<FragmentScanner> fragment_scanner;
    uint64_t unused_readahead_bytes;
    bool iterating_tasks = false;
    // Transitions to true when we've finished listing fragments for this task
    bool finished = false;
    // Transitions to true when we receieve an error or cancellation is requested
    bool canceled = false;
    uint32_t num_active_reads = 0;
    std::mutex task_mutex;
    ScanTaskPtr self;

    FragmentScanTask(std::unique_ptr<FragmentEvolutionStrategy> fragment_evolution,
                     std::shared_ptr<FragmentScanner> fragment_scanner,
                     uint64_t readahead_bytes)
        : fragment_evolution(std::move(fragment_evolution)),
          fragment_scanner(std::move(fragment_scanner)),
          unused_readahead_bytes(readahead_bytes) {}
  };

  struct AmountScanned {
    uint32_t num_reads = 0;
    uint64_t num_bytes = 0;
    bool finished = false;
  };

  Result<AmountScanned> DoScanMore(ScanTaskPtr scan_task, uint64_t readahead_bytes) {
    AmountScanned amount_scanned;
    bool has_data = scan_task->fragment_scanner->HasUnscannedData();
    while (has_data && amount_scanned.num_bytes < readahead_bytes) {
      ARROW_ASSIGN_OR_RAISE(Future<> read_tracker, plan_->BeginExternalTask());
      if (!read_tracker.is_valid()) {
        // Plan has been canceled.  Returning finished=true here will stop
        // any further reading of this file
        amount_scanned.finished = true;
        return amount_scanned;
      }
      // std::cout << "READ BEGIN\n";
      ReadTask next_read = scan_task->fragment_scanner->ScanMore(
          readahead_bytes - amount_scanned.num_bytes);
      uint64_t bytes_requested = next_read.num_bytes;
      amount_scanned.num_bytes += bytes_requested;
      amount_scanned.num_reads++;
      next_read.task
          .Then([this, scan_task,
                 bytes_requested](const std::shared_ptr<RecordBatch>& batch) {
            ARROW_RETURN_NOT_OK(
                plan_->ScheduleTask([this, batch, scan_task, bytes_requested]() {
                  this->ScanCallback(batch, bytes_requested, scan_task);
                  return Status::OK();
                }));
            return Status::OK();
          })
          .AddCallback([read_tracker](const Status&) mutable {
            // std::cout << "READ END\n";
            read_tracker.MarkFinished();
          });
      has_data = scan_task->fragment_scanner->HasUnscannedData();
    }
    amount_scanned.finished = !has_data;
    return amount_scanned;
  }

  // This is the "scan loop" which generates read tasks until
  // we have enough read tasks in flight to fill the readahead buffer
  // or we have run out of batches in the fragment.
  //
  // If we pause due to readahead then we will resume this loop later
  // when a batch is delivered (running the scan loop before processing
  // the batch)
  //
  // Eventually this loop will reach the end of the fragment at which point
  // it will schedule a task to process the next fragment in the queue (if there
  // is one) or mark the input finished (if there are no more fragments)
  Status ScanMore(ScanTaskPtr scan_task) {
    while (true) {
      uint64_t unused_readahead_bytes = 0;
      {
        std::lock_guard<std::mutex> lk(scan_task->task_mutex);
        if (scan_task->iterating_tasks) {
          return Status::OK();
        }
        scan_task->iterating_tasks = true;
        unused_readahead_bytes = scan_task->unused_readahead_bytes;
      }
      Result<AmountScanned> maybe_scanned = DoScanMore(scan_task, unused_readahead_bytes);
      {
        std::unique_lock<std::mutex> lk(scan_task->task_mutex);
        if (scan_task->canceled) {
          // An error already occurred, ignore the result and leave
          scan_task->iterating_tasks = false;
          return Status::OK();
        }
        DCHECK(!scan_task->finished);
        // Listing tasks failed.  Report the failure and leave
        if (!maybe_scanned.ok()) {
          scan_task->iterating_tasks = false;
          scan_task->canceled = true;
          scan_task->finished = true;
          lk.unlock();
          return maybe_scanned.status();
        }
        AmountScanned scanned = *maybe_scanned;
        // We should always generate at least one read task.  If this fails
        // we need to potentially mark the fragment finished in this loop.
        DCHECK_GT(scanned.num_reads, 0);
        scan_task->unused_readahead_bytes -= scanned.num_bytes;
        scan_task->num_active_reads += scanned.num_reads;
        if (scanned.finished) {
          // We reached the end of the file.
          scan_task->finished = true;
          scan_task->iterating_tasks = false;
          if (scan_task->num_active_reads == 0) {
            // Although a bit suprising, we can reach this spot if the read task(s)
            // quickly finished while we were listing them.
            ScannerSynchronization::NextFragment next_fragment =
                synchronization_.FinishFragmentAndGetNext();
            if (next_fragment.fragment) {
              ScanFragment(std::move(next_fragment.fragment));
            } else if (next_fragment.should_mark_input_finished) {
              batch_output_->InputFinished();
              finished_.MarkFinished();
            }
          }
          return Status::OK();
        }
        if (scan_task->unused_readahead_bytes < 0) {
          // We've started as many tasks as we can, time to leave
          scan_task->iterating_tasks = false;
          return Status::OK();
        }
        // If we get here then there is still room for more tasks (perhaps some tasks
        // finished while we were listing tasks) so we should list tasks again
      }
    }
  }

  void ScanCallback(const Result<std::shared_ptr<RecordBatch>>& batch,
                    uint64_t bytes_completed, ScanTaskPtr scan_task) {
    Status st = DoScanCallback(batch, bytes_completed, scan_task);
    if (!st.ok()) {
      std::unique_lock<std::mutex> lk(scan_task->task_mutex);
      if (!scan_task->canceled) {
        scan_task->canceled = true;
        lk.unlock();
        outputs_[0]->ErrorReceived(this, st);
        synchronization_.ReportFailedFragment(st);
      }
    }
  }

  Status DoScanCallback(const Result<std::shared_ptr<RecordBatch>>& batch,
                        uint64_t bytes_completed, ScanTaskPtr scan_task) {
    bool should_read_more = false;
    bool should_start_next_fragment = false;
    {
      std::lock_guard<std::mutex> lk(scan_task->task_mutex);
      if (scan_task->canceled) {
        return Status::OK();
      }
      ARROW_RETURN_NOT_OK(batch);
      scan_task->unused_readahead_bytes += bytes_completed;
      if (--scan_task->num_active_reads == 0 && scan_task->finished) {
        // If we've completed this fragment then we can start the next
        should_start_next_fragment = true;
      }
      // If there is space remaining then we should queue new reads before
      // we process this data
      should_read_more = scan_task->unused_readahead_bytes > 0 && !scan_task->finished &&
                         !scan_task->iterating_tasks;
    }
    if (should_read_more) {
      ARROW_RETURN_NOT_OK(ScanMore(scan_task));
    }
    ARROW_ASSIGN_OR_RAISE(compute::ExecBatch evolved_batch,
                          scan_task->fragment_evolution->EvolveBatch(*batch));
    // std::cout << "Delivering batch (should_start_next_fragment=" +
    //                  std::to_string(should_start_next_fragment) + ")\n";
    batch_output_->DeliverBatch(std::move(evolved_batch));
    if (should_start_next_fragment) {
      ScannerSynchronization::NextFragment next_fragment =
          synchronization_.FinishFragmentAndGetNext();
      if (next_fragment.fragment) {
        ScanFragment(std::move(next_fragment.fragment));
      } else if (next_fragment.should_mark_input_finished) {
        batch_output_->InputFinished();
        finished_.MarkFinished();
      }
    }
    return Status::OK();
  }

  Result<FragmentScanRequest> GetFragmentScanRequest(
      const Fragment& fragment, FragmentEvolutionStrategy* fragment_evolution) {
    FragmentScanRequest fragment_scan_request;
    ARROW_ASSIGN_OR_RAISE(fragment_scan_request.columns,
                          fragment_evolution->DevolveSelection(options_.columns));
    ARROW_ASSIGN_OR_RAISE(compute::Expression devolution_guarantee,
                          fragment_evolution->GetGuarantee(options_.columns));
    ARROW_ASSIGN_OR_RAISE(
        compute::Expression simplified_filter,
        compute::SimplifyWithGuarantee(options_.filter, devolution_guarantee));
    ARROW_ASSIGN_OR_RAISE(
        fragment_scan_request.filter,
        fragment_evolution->DevolveFilter(std::move(simplified_filter)));
    fragment_scan_request.format_scan_options = options_.format_options;
    return fragment_scan_request;
  }

  void ScanFragment(std::shared_ptr<Fragment> fragment) {
    // A callback that adds the fragment scan task to our list of tasks and bootstraps
    // scanning the task
    struct RegisterFragmentScanner {
      Status operator()(const std::shared_ptr<FragmentScanner>& fragment_scanner) {
        ScanTaskPtr scan_task;
        {
          std::lock_guard<std::mutex> active_scan_tasks_lock(self->mutex_);
          self->active_scan_tasks_.emplace_back(std::move(fragment_evolution),
                                                fragment_scanner,
                                                self->options_.target_bytes_readahead);
          scan_task = --self->active_scan_tasks_.end();
          scan_task->self = scan_task;
        }
        return self->ScanMore(scan_task);
      }

      ScanNode* self;
      std::unique_ptr<FragmentEvolutionStrategy> fragment_evolution;
    };

    Result<Future<>> maybe_scan_task = plan_->BeginExternalTask();
    if (!maybe_scan_task.ok() || !maybe_scan_task->is_valid()) {
      // Plan canceled, just return
      return;
    }
    // std::cout << "SCAN: BEGIN\n";
    Future<> scan_task = maybe_scan_task.MoveValueUnsafe();

    fragment->InspectFragment()
        .Then(
            [this, fragment](const std::shared_ptr<InspectedFragment>& inspected_fragment)
                -> Future<> {
              // Now that we have a fragment we need to use the dataset's evolution
              // strategy to figure out how to scan it
              std::unique_ptr<FragmentEvolutionStrategy> fragment_evolution =
                  options_.dataset->evolution_strategy()->GetStrategy(
                      *options_.dataset, *fragment, *inspected_fragment);
              ARROW_ASSIGN_OR_RAISE(
                  FragmentScanRequest fragment_scan_request,
                  GetFragmentScanRequest(*fragment, fragment_evolution.get()));

              return fragment->BeginScan(fragment_scan_request, *inspected_fragment)
                  .Then(RegisterFragmentScanner{this, std::move(fragment_evolution)});
            })
        .AddCallback([this, scan_task](const Status& st) mutable {
          if (!st.ok()) {
            // std::cout << "Reporting failed fragment: " + st.ToString() + "\n";
            outputs_[0]->ErrorReceived(this, st);
            synchronization_.ReportFailedFragment(st);
          }
          // std::cout << "SCAN: END\n";
          scan_task.MarkFinished();
        });
  }

  void QueueOrStartFragment(std::shared_ptr<Fragment> fragment) {
    if (synchronization_.QueueFragment(fragment)) {
      // std::cout << "ScanFragment\n";
      ScanFragment(std::move(fragment));
    }
  }

  ScanV2Options options_;
  ScannerSynchronization synchronization_;
  std::unique_ptr<BatchOutputStrategy> batch_output_;

  std::mutex mutex_;
  std::list<FragmentScanTask> active_scan_tasks_;
};

}  // namespace

namespace internal {
void InitializeScannerV2(arrow::compute::ExecFactoryRegistry* registry) {
  DCHECK_OK(registry->AddFactory("scan2", ScanNode::Make));
}
}  // namespace internal
}  // namespace dataset
}  // namespace arrow