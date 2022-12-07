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

#include "arrow/compute/exec/groupby.h"

#include <mutex>
#include <thread>
#include <unordered_map>

#include "arrow/compute/exec_internal.h"
#include "arrow/compute/registry.h"
#include "arrow/compute/row/grouper.h"
#include "arrow/record_batch.h"
#include "arrow/table.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/logging.h"
#include "arrow/util/string.h"
#include "arrow/util/task_group.h"

namespace arrow {

using internal::ToChars;

namespace compute {

namespace {

std::shared_ptr<Schema> SimpleSchemaForBatch(const ExecBatch& batch, int num_keys) {
  std::vector<std::shared_ptr<Field>> fields;
  for (int i = 0; i < batch.num_values(); i++) {
    std::string name = (i < num_keys) ? "key_" + ::arrow::internal::ToChars(i)
                                      : "segment_" + ::arrow::internal::ToChars(i);
    fields.push_back(field(name, batch.values[i].type()));
  }
  return schema(std::move(fields));
}

}  // namespace

Result<std::shared_ptr<Table>> GroupBy(
    const std::vector<std::shared_ptr<Array>>& arguments,
    const std::vector<std::shared_ptr<Array>>& keys,
    const std::vector<std::shared_ptr<Array>>& segment_keys,
    const std::vector<SimpleAggregate>& aggregates, bool use_threads, ExecContext* ctx) {
  if (arguments.size() != aggregates.size()) {
    return Status::Invalid("arguments and aggregates must be the same size");
  }

  if (arguments.empty() && keys.empty()) {
    return Table::MakeEmpty(schema({}));
  }

  std::vector<Datum> all_columns;
  int64_t length = 0;
  for (const auto& key : keys) {
    if (length == 0) {
      length = key->length();
    } else {
      if (length != key->length()) {
        return Status::Invalid("all inputs to GroupBy must have the same length");
      }
    }
    all_columns.emplace_back(key);
  }
  for (const auto& argument : arguments) {
    if (length == 0) {
      length = argument->length();
    } else {
      if (length != argument->length()) {
        return Status::Invalid("all inputs to GroupBy must have the same length");
      }
    }
  }
  // Segments aren't used to calculate the length so we add them after one pass
  // through arguments
  for (const auto& segment : segment_keys) {
    if (length != segment->length()) {
      return Status::Invalid("all inputs to GroupBy must have the same length");
    }
    all_columns.emplace_back(segment);
  }
  // Second pass through arguments so our all_columns will be {keys, segment_keys, args}
  for (const auto& argument : arguments) {
    all_columns.emplace_back(argument);
  }
  ExecBatch input_batch(std::move(all_columns), length);
  std::shared_ptr<Schema> batch_schema =
      SimpleSchemaForBatch(input_batch, static_cast<int>(keys.size()));
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<RecordBatch> rb,
                        input_batch.ToRecordBatch(std::move(batch_schema)));
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Table> table, Table::FromRecordBatches({rb}));

  std::vector<FieldRef> key_refs;
  for (int i = 0; i < static_cast<int>(keys.size()); i++) {
    key_refs.emplace_back(i);
  }
  std::vector<FieldRef> segment_refs;
  for (int i = 0; i < static_cast<int>(segment_keys.size()); i++) {
    segment_refs.emplace_back(static_cast<int>(i + keys.size()));
  }

  std::vector<Aggregate> plan_aggregates;
  for (std::size_t i = 0; i < aggregates.size(); i++) {
    const SimpleAggregate& agg = aggregates[i];
    std::string output_field_name;
    if (agg.function.size() > 5 && agg.function.substr(0, 5) == "hash_") {
      if (keys.empty()) {
        return Status::Invalid(
            "You cannot use the aggregate function ", agg.function,
            " if there are no keys.  Use the variant without hash_ instead");
      }
      output_field_name = agg.function.substr(5);
    } else {
      if (!keys.empty()) {
        return Status::Invalid(
            "You cannot use the aggregate function ", agg.function,
            " if there are keys.  Use the variant starting with hash_ instead");
      }
      output_field_name = agg.function;
    }
    plan_aggregates.push_back(
        {agg.function, agg.options,
         /*target=*/{static_cast<int>(i + keys.size() + segment_keys.size())},
         output_field_name});
  }

  Declaration plan = Declaration::Sequence(
      {{"table_source", TableSourceNodeOptions(std::move(table))},
       {"aggregate", AggregateNodeOptions(std::move(plan_aggregates), std::move(key_refs),
                                          std::move(segment_refs))}});

  return DeclarationToTable(plan);
}

}  // namespace compute
}  // namespace arrow
