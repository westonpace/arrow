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

#include <sstream>

#include "arrow/compute/api_vector.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/exec/accumulation_queue.h"
#include "arrow/compute/exec/exec_plan.h"
#include "arrow/compute/exec/expression.h"
#include "arrow/compute/exec/map_node.h"
#include "arrow/compute/exec/options.h"
#include "arrow/compute/exec/query_context.h"
#include "arrow/compute/exec/util.h"
#include "arrow/datum.h"
#include "arrow/result.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/future.h"
#include "arrow/util/logging.h"
#include "arrow/util/tracing_internal.h"

namespace arrow {

using internal::checked_cast;

namespace compute {
namespace {

class FetchCounter {
 public:
  struct Page {
    int64_t to_skip;
    int64_t to_send;
  };

  FetchCounter(int64_t rows_to_send, int64_t rows_to_skip)
      : rows_to_send_(rows_to_send), rows_to_skip_(rows_to_skip) {}

  Page NextPage(const ExecBatch& batch) {
    int64_t rows_in_batch_to_skip = 0;
    if (rows_to_skip_ > 0) {
      rows_in_batch_to_skip = std::min(rows_to_skip_, batch.length);
      rows_to_skip_ -= rows_in_batch_to_skip;
    }

    int64_t rows_in_batch_to_send = 0;
    if (rows_to_send_ > 0) {
      rows_in_batch_to_send =
          std::min(rows_to_send_, batch.length - rows_in_batch_to_skip);
      rows_to_send_ -= rows_in_batch_to_send;
    }
    return {rows_in_batch_to_skip, rows_in_batch_to_send};
  }

 private:
  int64_t rows_to_send_;
  int64_t rows_to_skip_;
};

class FetchNode : public MapNode, public TracedNode<FetchNode> {
 public:
  FetchNode(ExecPlan* plan, std::vector<ExecNode*> inputs,
            std::shared_ptr<Schema> output_schema, int64_t offset, int64_t count)
      : MapNode(plan, std::move(inputs), std::move(output_schema)),
        max_to_send_(count),
        fetch_counter_(offset, count) {
    sequencing_queue_ = util::SequencingQueue::Make(
        [this](ExecBatch batch) { return SequenceProcess(std::move(batch)); },
        [this](util::SequencingQueue::Task task) {
          return SequenceSchedule(std::move(task));
        });
  }

  static Result<ExecNode*> Make(ExecPlan* plan, std::vector<ExecNode*> inputs,
                                const ExecNodeOptions& options) {
    RETURN_NOT_OK(ValidateExecNodeInputs(plan, inputs, 1, "FetchNode"));

    const auto& fetch_options = checked_cast<const FetchNodeOptions&>(options);

    int64_t offset = fetch_options.offset;
    int64_t count = fetch_options.count;

    if (offset < 0) {
      return Status::Invalid("`offset` must be non-negative");
    }
    if (count < 0) {
      return Status::Invalid("`count` must be non-negative");
    }

    std::shared_ptr<Schema> output_schema = inputs[0]->output_schema();
    return plan->EmplaceNode<FetchNode>(plan, std::move(inputs), std::move(output_schema),
                                        offset, count);
  }

  const char* kind_name() const override { return "FetchNode"; }

  Result<ExecBatch> ProcessBatch(ExecBatch batch) override {
    std::vector<Datum> values{exprs_.size()};
    for (size_t i = 0; i < exprs_.size(); ++i) {
      util::tracing::Span span;
      START_COMPUTE_SPAN(span, "Project",
                         {{"project.type", exprs_[i].type()->ToString()},
                          {"project.length", batch.length},
                          {"project.expression", exprs_[i].ToString()}});
      ARROW_ASSIGN_OR_RAISE(Expression simplified_expr,
                            SimplifyWithGuarantee(exprs_[i], batch.guarantee));

      ARROW_ASSIGN_OR_RAISE(
          values[i], ExecuteScalarExpression(simplified_expr, batch,
                                             plan()->query_context()->exec_context()));
    }
    return ExecBatch{std::move(values), batch.length};
  }

  Result<std::optional<util::SequencingQueue::Task>> SequenceProcess(ExecBatch batch) {
    FetchCounter::Page page = fetch_counter_.NextPage(batch);
    if (page.to_send > 0) {
      ExecBatch batch_to_send = std::move(batch);
      if (page.to_skip > 0) {
        batch_to_send = batch_to_send.Slice(0, page.to_skip);
      }
      return [this, batch_to_send]() mutable {
        return output_->InputReceived(this, std::move(batch_to_send));
      };
    } else if (!source_stopped_) {
      source_stopped_ = true;
      inputs_[0]->StopProducing();
      output_->InputFinished()
    }
    return std::nullopt;
  }

  void SequenceSchedule(util::SequencingQueue::Task task) {
    plan_->query_context()->ScheduleTask(std::move(task), "FetchNode::ProcessBatch");
  }

 protected:
  std::string ToStringExtra(int indent = 0) const override {
    std::stringstream ss;
    ss << "projection=[";
    for (int i = 0; static_cast<size_t>(i) < exprs_.size(); i++) {
      if (i > 0) ss << ", ";
      auto repr = exprs_[i].ToString();
      if (repr != output_schema_->field(i)->name()) {
        ss << '"' << output_schema_->field(i)->name() << "\": ";
      }
      ss << repr;
    }
    ss << ']';
    return ss.str();
  }

 private:
  bool source_stopped_ = false;
  int64_t max_to_send_;
  FetchCounter fetch_counter_;
  std::unique_ptr<util::SequencingQueue> sequencing_queue_;
};

}  // namespace

namespace internal {

void RegisterProjectNode(ExecFactoryRegistry* registry) {
  DCHECK_OK(registry->AddFactory("project", ProjectNode::Make));
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
