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

class FetchNode : public MapNode, public TracedNode<FetchNode> {
 public:
  FetchNode(ExecPlan* plan, std::vector<ExecNode*> inputs,
            std::shared_ptr<Schema> output_schema, int64_t offset, int64_t count)
      : MapNode(plan, std::move(inputs), std::move(output_schema)),
        offset_(offset),
        count_(count) {}

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

  const char* kind_name() const override { return "ProjectNode"; }

  Result<ExecBatch> DoProject(const ExecBatch& target) {
    std::vector<Datum> values{exprs_.size()};
    for (size_t i = 0; i < exprs_.size(); ++i) {
      util::tracing::Span span;
      START_COMPUTE_SPAN(span, "Project",
                         {{"project.type", exprs_[i].type()->ToString()},
                          {"project.length", target.length},
                          {"project.expression", exprs_[i].ToString()}});
      ARROW_ASSIGN_OR_RAISE(Expression simplified_expr,
                            SimplifyWithGuarantee(exprs_[i], target.guarantee));

      ARROW_ASSIGN_OR_RAISE(
          values[i], ExecuteScalarExpression(simplified_expr, target,
                                             plan()->query_context()->exec_context()));
    }
    return ExecBatch{std::move(values), target.length};
  }

  void InputReceived(ExecNode* input, ExecBatch batch) override {
    DCHECK_EQ(input, inputs_[0]);
    auto func = [this](ExecBatch batch) {
      auto result = DoProject(std::move(batch));
      return result;
    };
    this->SubmitTask(std::move(func), std::move(batch));
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
  int64_t offset_;
  int64_t count_;
};

}  // namespace

namespace internal {

void RegisterProjectNode(ExecFactoryRegistry* registry) {
  DCHECK_OK(registry->AddFactory("project", ProjectNode::Make));
}

}  // namespace internal
}  // namespace compute
}  // namespace arrow
