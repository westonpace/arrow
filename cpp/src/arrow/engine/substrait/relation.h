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

#include "arrow/compute/exec/exec_plan.h"
#include "arrow/type_fwd.h"

namespace arrow {
namespace engine {

/// Execution information resulting from converting a Substrait relation.
struct ARROW_ENGINE_EXPORT DeclarationInfo {
  /// The compute declaration produced thus far.
  compute::Declaration declaration;

  std::shared_ptr<Schema> output_schema;
};

/// Information resulting from converting a Substrait relation.
///
/// RelationInfo adds the "output indices" field for the extension to define how the
/// fields should be mapped to get the standard indices expected by Substrait.
struct ARROW_ENGINE_EXPORT RelationInfo {
  /// The execution information produced thus far.
  DeclarationInfo decl_info;
  /// The total number of input fields across all inputs of the relation.
  int total_input_fields;
  /// A vector of indices, one per input field per input in order, each index referring
  /// to the corresponding field within the output schema, if it is in the output, or -1
  /// otherwise. Each location in this vector is a field input index. This vector is
  /// useful for translating selected field input indices (often from an output mapping in
  /// a Substrait plan) of a join-type relation to their locations in the output schema of
  /// the relation. This vector is undefined if the translation is unsupported, i.e., when
  /// there is an output field that does not have an input field corresponding to it. When
  /// defined, the size of this vector is equal to `total_input_fields`.
  std::optional<std::vector<int>> field_output_indices;
};

}  // namespace engine
}  // namespace arrow
