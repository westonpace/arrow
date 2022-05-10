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

#include "arrow/dataset/dataset.h"

#include <memory>
#include <utility>

#include "arrow/dataset/dataset_internal.h"
#include "arrow/dataset/scanner.h"
#include "arrow/table.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/byte_size.h"
#include "arrow/util/iterator.h"
#include "arrow/util/logging.h"
#include "arrow/util/make_unique.h"

namespace arrow {

using internal::checked_pointer_cast;

namespace dataset {

Fragment::Fragment(compute::Expression partition_expression,
                   std::shared_ptr<Schema> physical_schema)
    : partition_expression_(std::move(partition_expression)),
      physical_schema_(std::move(physical_schema)) {}

Future<std::shared_ptr<InspectedFragment>> Fragment::InspectFragment() {
  return Status::NotImplemented("Inspect fragment");
}

Future<std::shared_ptr<FragmentScanner>> Fragment::BeginScan(
    const FragmentScanRequest& request, const InspectedFragment& inspected_fragment) {
  return Status::NotImplemented("New scan method");
}

Result<std::shared_ptr<Schema>> Fragment::ReadPhysicalSchema() {
  {
    auto lock = physical_schema_mutex_.Lock();
    if (physical_schema_ != nullptr) return physical_schema_;
  }

  // allow ReadPhysicalSchemaImpl to lock mutex_, if necessary
  ARROW_ASSIGN_OR_RAISE(auto physical_schema, ReadPhysicalSchemaImpl());

  auto lock = physical_schema_mutex_.Lock();
  if (physical_schema_ == nullptr) {
    physical_schema_ = std::move(physical_schema);
  }
  return physical_schema_;
}

Future<util::optional<int64_t>> Fragment::CountRows(compute::Expression,
                                                    const std::shared_ptr<ScanOptions>&) {
  return Future<util::optional<int64_t>>::MakeFinished(util::nullopt);
}

Result<std::shared_ptr<Schema>> InMemoryFragment::ReadPhysicalSchemaImpl() {
  return physical_schema_;
}

InMemoryFragment::InMemoryFragment(std::shared_ptr<Schema> schema,
                                   RecordBatchVector record_batches,
                                   compute::Expression partition_expression)
    : Fragment(std::move(partition_expression), std::move(schema)),
      record_batches_(std::move(record_batches)) {
  DCHECK_NE(physical_schema_, nullptr);
}

InMemoryFragment::InMemoryFragment(RecordBatchVector record_batches,
                                   compute::Expression partition_expression)
    : Fragment(std::move(partition_expression), /*schema=*/nullptr),
      record_batches_(std::move(record_batches)) {
  // Order of argument evaluation is undefined, so compute physical_schema here
  physical_schema_ = record_batches_.empty() ? schema({}) : record_batches_[0]->schema();
}

Result<RecordBatchGenerator> InMemoryFragment::ScanBatchesAsync(
    const std::shared_ptr<ScanOptions>& options) {
  struct State {
    State(std::shared_ptr<InMemoryFragment> fragment, int64_t batch_size)
        : fragment(std::move(fragment)),
          batch_index(0),
          offset(0),
          batch_size(batch_size) {}

    std::shared_ptr<RecordBatch> Next() {
      const auto& next_parent = fragment->record_batches_[batch_index];
      if (offset < next_parent->num_rows()) {
        auto next = next_parent->Slice(offset, batch_size);
        offset += batch_size;
        return next;
      }
      batch_index++;
      offset = 0;
      return nullptr;
    }

    bool Finished() { return batch_index >= fragment->record_batches_.size(); }

    std::shared_ptr<InMemoryFragment> fragment;
    std::size_t batch_index;
    int64_t offset;
    int64_t batch_size;
  };

  struct Generator {
    Generator(std::shared_ptr<InMemoryFragment> fragment, int64_t batch_size)
        : state(std::make_shared<State>(std::move(fragment), batch_size)) {}

    Future<std::shared_ptr<RecordBatch>> operator()() {
      while (!state->Finished()) {
        auto next = state->Next();
        if (next) {
          return Future<std::shared_ptr<RecordBatch>>::MakeFinished(std::move(next));
        }
      }
      return AsyncGeneratorEnd<std::shared_ptr<RecordBatch>>();
    }

    std::shared_ptr<State> state;
  };
  return Generator(checked_pointer_cast<InMemoryFragment>(shared_from_this()),
                   options->batch_size);
}

Future<util::optional<int64_t>> InMemoryFragment::CountRows(
    compute::Expression predicate, const std::shared_ptr<ScanOptions>& options) {
  if (ExpressionHasFieldRefs(predicate)) {
    return Future<util::optional<int64_t>>::MakeFinished(util::nullopt);
  }
  int64_t total = 0;
  for (const auto& batch : record_batches_) {
    total += batch->num_rows();
  }
  return Future<util::optional<int64_t>>::MakeFinished(total);
}

Future<std::shared_ptr<InspectedFragment>> InMemoryFragment::InspectFragment() {
  return std::make_shared<InspectedFragment>(physical_schema_->field_names());
}

class InMemoryFragment::Scanner : public FragmentScanner {
 public:
  Scanner(InMemoryFragment* fragment) : fragment_(fragment) {}

  virtual ReadTask ScanMore(uint64_t target_num_bytes) {
    const std::shared_ptr<RecordBatch>& next_batch = fragment_->record_batches_[index++];
    int64_t actual_bytes = arrow::util::TotalBufferSize(*next_batch);
    return ReadTask{Future<std::shared_ptr<RecordBatch>>::MakeFinished(next_batch),
                    static_cast<uint64_t>(actual_bytes)};
  }
  virtual bool HasUnscannedData() {
    return index < static_cast<int>(fragment_->record_batches_.size());
  }

 private:
  InMemoryFragment* fragment_;
  int index = 0;
};

Future<std::shared_ptr<FragmentScanner>> InMemoryFragment::BeginScan(
    const FragmentScanRequest& request, const InspectedFragment& inspected_fragment) {
  return Future<std::shared_ptr<FragmentScanner>>::MakeFinished(
      std::make_shared<InMemoryFragment::Scanner>(this));
}

Dataset::Dataset(std::shared_ptr<Schema> schema, compute::Expression partition_expression)
    : schema_(std::move(schema)),
      partition_expression_(std::move(partition_expression)) {}

Result<std::shared_ptr<ScannerBuilder>> Dataset::NewScan() {
  return std::make_shared<ScannerBuilder>(this->shared_from_this());
}

Result<FragmentIterator> Dataset::GetFragments() {
  return GetFragments(compute::literal(true));
}

Result<FragmentIterator> Dataset::GetFragments(compute::Expression predicate) {
  ARROW_ASSIGN_OR_RAISE(
      predicate, SimplifyWithGuarantee(std::move(predicate), partition_expression_));
  return predicate.IsSatisfiable() ? GetFragmentsImpl(std::move(predicate))
                                   : MakeEmptyIterator<std::shared_ptr<Fragment>>();
}

struct VectorRecordBatchGenerator : InMemoryDataset::RecordBatchGenerator {
  explicit VectorRecordBatchGenerator(RecordBatchVector batches)
      : batches_(std::move(batches)) {}

  RecordBatchIterator Get() const final { return MakeVectorIterator(batches_); }

  RecordBatchVector batches_;
};

InMemoryDataset::InMemoryDataset(std::shared_ptr<Schema> schema,
                                 RecordBatchVector batches)
    : Dataset(std::move(schema)),
      get_batches_(new VectorRecordBatchGenerator(std::move(batches))) {}

struct TableRecordBatchGenerator : InMemoryDataset::RecordBatchGenerator {
  explicit TableRecordBatchGenerator(std::shared_ptr<Table> table)
      : table_(std::move(table)) {}

  RecordBatchIterator Get() const final {
    auto reader = std::make_shared<TableBatchReader>(*table_);
    auto table = table_;
    return MakeFunctionIterator([reader, table] { return reader->Next(); });
  }

  std::shared_ptr<Table> table_;
};

InMemoryDataset::InMemoryDataset(std::shared_ptr<Table> table)
    : Dataset(table->schema()),
      get_batches_(new TableRecordBatchGenerator(std::move(table))) {}

Result<std::shared_ptr<Dataset>> InMemoryDataset::ReplaceSchema(
    std::shared_ptr<Schema> schema) const {
  RETURN_NOT_OK(CheckProjectable(*schema_, *schema));
  return std::make_shared<InMemoryDataset>(std::move(schema), get_batches_);
}

Result<FragmentIterator> InMemoryDataset::GetFragmentsImpl(compute::Expression) {
  auto schema = this->schema();

  auto create_fragment =
      [schema](std::shared_ptr<RecordBatch> batch) -> Result<std::shared_ptr<Fragment>> {
    RETURN_NOT_OK(CheckProjectable(*schema, *batch->schema()));
    return std::make_shared<InMemoryFragment>(RecordBatchVector{std::move(batch)});
  };

  auto batches_it = get_batches_->Get();
  return MakeMaybeMapIterator(std::move(create_fragment), std::move(batches_it));
}

Result<std::shared_ptr<UnionDataset>> UnionDataset::Make(std::shared_ptr<Schema> schema,
                                                         DatasetVector children) {
  for (const auto& child : children) {
    if (!child->schema()->Equals(*schema)) {
      return Status::TypeError("child Dataset had schema ", *child->schema(),
                               " but the union schema was ", *schema);
    }
  }

  return std::shared_ptr<UnionDataset>(
      new UnionDataset(std::move(schema), std::move(children)));
}

Result<std::shared_ptr<Dataset>> UnionDataset::ReplaceSchema(
    std::shared_ptr<Schema> schema) const {
  auto children = children_;
  for (auto& child : children) {
    ARROW_ASSIGN_OR_RAISE(child, child->ReplaceSchema(schema));
  }

  return std::shared_ptr<Dataset>(
      new UnionDataset(std::move(schema), std::move(children)));
}

Result<FragmentIterator> UnionDataset::GetFragmentsImpl(compute::Expression predicate) {
  return GetFragmentsFromDatasets(children_, predicate);
}

namespace {

class BasicFragmentEvolution : public FragmentEvolutionStrategy {
 public:
  BasicFragmentEvolution(std::vector<int> ds_to_frag_map, Schema* dataset_schema)
      : ds_to_frag_map(std::move(ds_to_frag_map)), dataset_schema(dataset_schema) {}

  virtual Result<compute::Expression> GetGuarantee(
      const std::vector<FieldPath>& dataset_schema_selection) const override {
    std::vector<compute::Expression> missing_fields;
    for (const FieldPath& path : dataset_schema_selection) {
      int top_level_field_idx = path[0];
      if (ds_to_frag_map[top_level_field_idx] < 0) {
        missing_fields.push_back(compute::equal(
            compute::field_ref(top_level_field_idx),
            compute::literal(
                MakeNullScalar(dataset_schema->fields()[top_level_field_idx]->type()))));
      }
    }
    if (missing_fields.empty()) {
      return compute::literal(true);
    }
    if (missing_fields.size() == 1) {
      return missing_fields[0];
    }
    return compute::and_(missing_fields);
  }

  Result<std::vector<FragmentSelectionColumn>> DevolveSelection(
      const std::vector<FieldPath>& dataset_schema_selection) const override {
    std::vector<FragmentSelectionColumn> desired_columns;
    for (const FieldPath& path : dataset_schema_selection) {
      int top_level_field_idx = path[0];
      int dest_top_level_idx = ds_to_frag_map[top_level_field_idx];
      if (dest_top_level_idx > 0) {
        ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Field> field, path.Get(*dataset_schema));
        std::vector<int> dest_path_indices(path.indices());
        dest_path_indices[0] = dest_top_level_idx;
        desired_columns.push_back(
            FragmentSelectionColumn{FieldPath(dest_path_indices), field->type().get()});
      }
    }
    return std::move(desired_columns);
  };

  virtual Result<compute::Expression> DevolveFilter(
      const compute::Expression& filter) const override {
    return filter;
  };

  virtual Result<compute::ExecBatch> EvolveBatch(
      const std::shared_ptr<RecordBatch>& batch) const override {
    std::vector<Datum> columns;
    for (int idx : ds_to_frag_map) {
      if (idx < 0) {
        columns.push_back(MakeNullScalar(dataset_schema->field(idx)->type()));
      } else {
        columns.push_back(batch->column(idx));
      }
    }
    return compute::ExecBatch(columns, batch->num_rows());
  };

  std::string ToString() const override { return "basic-fragment-evolution"; }

  std::vector<int> ds_to_frag_map;
  Schema* dataset_schema;

  static std::unique_ptr<BasicFragmentEvolution> Make(
      const std::shared_ptr<Schema>& dataset_schema,
      const std::vector<std::string>& fragment_column_names) {
    std::vector<int> ds_to_frag_map;
    std::unordered_map<std::string, int> column_names_map;
    for (size_t i = 0; i < fragment_column_names.size(); i++) {
      column_names_map[fragment_column_names[i]] = static_cast<int>(i);
    }
    for (int idx = 0; idx < dataset_schema->num_fields(); idx++) {
      const std::string& field_name = dataset_schema->field(idx)->name();
      auto column_idx_itr = column_names_map.find(field_name);
      if (column_idx_itr == column_names_map.end()) {
        ds_to_frag_map.push_back(-1);
      } else {
        ds_to_frag_map.push_back(column_idx_itr->second);
      }
    }
    return ::arrow::internal::make_unique<BasicFragmentEvolution>(
        std::move(ds_to_frag_map), dataset_schema.get());
  }
};

class BasicDatasetEvolutionStrategy : public DatasetEvolutionStrategy {
  std::unique_ptr<FragmentEvolutionStrategy> GetStrategy(
      const Dataset& dataset, const Fragment& fragment,
      const InspectedFragment& inspected_fragment) override {
    return BasicFragmentEvolution::Make(dataset.schema(),
                                        inspected_fragment.column_names);
  }

  std::string ToString() const override { return "basic-dataset-evolution"; }
};

}  // namespace

std::unique_ptr<DatasetEvolutionStrategy> MakeBasicDatasetEvolutionStrategy() {
  return ::arrow::internal::make_unique<BasicDatasetEvolutionStrategy>();
}

}  // namespace dataset
}  // namespace arrow
