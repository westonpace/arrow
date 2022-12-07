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

#include <memory>
#include <vector>

#include "arrow/compute/api_aggregate.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/exec/exec_plan.h"
#include "arrow/compute/kernel.h"
#include "arrow/datum.h"
#include "arrow/result.h"
#include "arrow/util/visibility.h"

namespace arrow {
namespace compute {

/// Simplified aggregate specifier that does not need a target
///
/// This is used for \see GroupBy which applies a single aggregate to each
/// input column.
struct SimpleAggregate {
  std::string function;
  std::shared_ptr<FunctionOptions> options;
};

/// Convenience function to perform a group-by given the arguments and keys as columns
///
/// The result will be calculated using an exec plan with an aggregate node
///
/// The output may contain multiple chunks if the input is large enough to be
/// processed in pieces
///
/// The aggregates vector must be the same size as the arguments vector.  Each aggregate
/// will be applied to the argument with the same index.  An array could be included in
/// the `arguments` vector multiple times to compute multiple aggregates.
///
/// \return a table that will have one column for each aggregate, named after they
/// aggregate function, and one column for each key, named key0, key1, ...
///
/// If there are no arguments/aggregates then the returned table will have one row
/// for each unique combination of keys
///
/// If there are no keys then the aggregates will be applied to the full array
ARROW_EXPORT
Result<std::shared_ptr<Table>> GroupBy(
    const std::vector<std::shared_ptr<Array>>& arguments,
    const std::vector<std::shared_ptr<Array>>& keys,
    const std::vector<std::shared_ptr<Array>>& segments,
    const std::vector<SimpleAggregate>& aggregates, bool use_threads = false,
    ExecContext* ctx = default_exec_context());

}  // namespace compute
}  // namespace arrow
