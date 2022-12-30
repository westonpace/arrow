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

#include "arrow/compute/exec/map_node.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/compute/exec.h"
#include "arrow/compute/exec/expression.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/logging.h"
#include "arrow/util/tracing_internal.h"

namespace arrow {
namespace compute {

MapNode::MapNode(ExecPlan* plan, std::vector<ExecNode*> inputs,
                 std::shared_ptr<Schema> output_schema)
    : ExecNode(plan, std::move(inputs), /*input_labels=*/{"target"},
               std::move(output_schema),
               /*num_outputs=*/1) {}

void MapNode::ErrorReceived(ExecNode* input, Status error) {
  DCHECK_EQ(input, inputs_[0]);
  EVENT(span_, "ErrorReceived", {{"error.message", error.message()}});
  outputs_[0]->ErrorReceived(this, std::move(error));
}

void MapNode::InputFinished(ExecNode* input, int total_batches) {
  DCHECK_EQ(input, inputs_[0]);
  EVENT(span_, "InputFinished", {{"batches.length", total_batches}});
  outputs_[0]->InputFinished(this, total_batches);
  if (input_counter_.SetTotal(total_batches)) {
    this->Finish();
  }
}

Status MapNode::StartProducing() {
  START_COMPUTE_SPAN(
      span_, std::string(kind_name()) + ":" + label(),
      {{"node.label", label()}, {"node.detail", ToString()}, {"node.kind", kind_name()}});
  return Status::OK();
}

void MapNode::PauseProducing(ExecNode* output, int32_t counter) {
  inputs_[0]->PauseProducing(this, counter);
}

void MapNode::ResumeProducing(ExecNode* output, int32_t counter) {
  inputs_[0]->ResumeProducing(this, counter);
}

void MapNode::StopProducing(ExecNode* output) {
  DCHECK_EQ(output, outputs_[0]);
  input_counter_.Cancel();
  inputs_[0]->StopProducing(this);
}

void MapNode::StopProducing() {
  input_counter_.Cancel();
  EVENT(span_, "StopProducing");
}

void MapNode::MapBatch(std::function<Result<ExecBatch>(ExecBatch)> map_fn,
                       ExecBatch batch) {
  auto guarantee = batch.guarantee;
  auto output_batch = map_fn(std::move(batch));
  if (ErrorIfNotOk(output_batch.status())) {
    input_counter_.Cancel();
    return;
  }
  output_batch->guarantee = guarantee;
  outputs_[0]->InputReceived(this, output_batch.MoveValueUnsafe());
  if (input_counter_.Increment()) {
    this->Finish();
  }
}

void MapNode::Finish() {}

}  // namespace compute
}  // namespace arrow
