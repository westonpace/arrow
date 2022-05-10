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

// This API is EXPERIMENTAL.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "arrow/compute/exec/expression.h"
#include "arrow/dataset/type_fwd.h"
#include "arrow/dataset/visibility.h"
#include "arrow/util/future.h"
#include "arrow/util/macros.h"
#include "arrow/util/mutex.h"
#include "arrow/util/optional.h"

namespace arrow {
namespace dataset {

using RecordBatchGenerator = std::function<Future<std::shared_ptr<RecordBatch>>()>;

/// \brief Description of a column to scan
struct FragmentSelectionColumn {
  /// \brief The path to the column to load
  FieldPath column_reference;
  /// \brief The requested type of the column
  ///
  /// A format may choose to ignore this field completely.  For example, when
  /// reading from IPC the reader can just return the column in the data type
  /// that is stored on disk.  In this case the scan node will cast the column
  /// to the appropriate type later.
  ///
  /// However, some formats may be capable of casting on the fly.  For example,
  /// when reading from CSV, if we know the target type of the column, we can
  /// convert from string to the target type as we read.
  DataType* requested_type;
};
/// \brief Instructions for scanning a particular fragment
///
/// The fragment scan request is dervied from ScanV2Options.  The main
/// difference is that the scan options are based on the dataset schema
/// while the fragment request is based on the fragment schema.
struct FragmentScanRequest {
  /// \brief A row filter
  ///
  /// The filter expression should be written against the fragment schema.
  ///
  /// \see ScanV2Options for details on how this filter should be applied
  compute::Expression filter = compute::literal(true);

  /// \brief The columns to scan
  ///
  /// These indices refer to the fragment schema
  ///
  /// Note: This is NOT a simple list of top-level column indices.
  /// For more details \see ScanV2Options
  ///
  /// If possible a fragment should only read from disk the data needed
  /// to satisfy these columns.  If a format cannot partially read a nested
  /// column (e.g. JSON) then it must apply the column selection (in memory)
  /// before returning the scanned batch.
  std::vector<FragmentSelectionColumn> columns;
  /// \brief Options specific to the format being scanned
  FragmentScanOptions* format_scan_options;
};

/// \brief A collection of reads issued by a FragmentScanner
struct ReadTask {
  /// \brief A future which will complete when the read is finished
  Future<std::shared_ptr<RecordBatch>> task;
  /// \brief An estimate of the number of data bytes that will
  ///        be read when all the reads have finished
  ///
  /// This may be lower than the requested number of data bytes
  /// if the fragment is almost finished
  ///
  /// This may be higher than the requested number of data bytes
  /// if the fragment had to read more to respect record batch
  /// boundaries
  uint64_t num_bytes;
};

class FragmentScanner {
 public:
  /// This instance will only be destroyed after all ongoing scan futures
  /// have been completed.
  ///
  /// This means any callbacks created as part of the scan can safely
  /// capture `this`
  virtual ~FragmentScanner() = default;
  /// \brief Scan further ahead into the fragment
  /// \param target_num_bytes The number of data bytes the scanner is ready to receive
  /// \return Futures for each read operation that has been started
  ///
  /// `target_num_bytes` is a hint only and fragments do not need to match this
  /// exactly.  For example, when scanning a parquet file, reads have to align
  /// with pages and so it is often neccesary to read more than asked for.
  /// See ScanV2Options for more details.
  ///
  /// It is acceptable for a future to complete with an empty batch.
  ///
  /// This may be called before the previous read task has finished and the caller
  /// should always check HasUnscannedData before each call.
  virtual ReadTask ScanMore(uint64_t target_num_bytes) = 0;

  /// \brief True if there is still data that has not been scanned
  ///
  /// This should return true when there is no more data to start scanning.
  /// Some scan operations may still be in progress and that is fine.
  virtual bool HasUnscannedData() = 0;
};

struct InspectedFragment {
  InspectedFragment(std::vector<std::string> column_names)
      : column_names(std::move(column_names)) {}
  std::vector<std::string> column_names;
};

/// \brief A granular piece of a Dataset, such as an individual file.
///
/// A Fragment can be read/scanned separately from other fragments. It yields a
/// collection of RecordBatches when scanned
///
/// Note that Fragments have well defined physical schemas which are reconciled by
/// the Datasets which contain them; these physical schemas may differ from a parent
/// Dataset's schema and the physical schemas of sibling Fragments.
class ARROW_DS_EXPORT Fragment : public std::enable_shared_from_this<Fragment> {
 public:
  /// \brief Return the physical schema of the Fragment.
  ///
  /// The physical schema is also called the writer schema.
  /// This method is blocking and may suffer from high latency filesystem.
  /// The schema is cached after being read once, or may be specified at construction.
  Result<std::shared_ptr<Schema>> ReadPhysicalSchema();

  /// An asynchronous version of Scan
  virtual Result<RecordBatchGenerator> ScanBatchesAsync(
      const std::shared_ptr<ScanOptions>& options) = 0;

  /// \brief Inspect a fragment to learn basic information
  ///
  /// This will be called before a scan and a fragment should attach whatever
  /// information will be needed to figure out an evolution strategy.  This information
  /// will then be passed to the call to BeginScan
  virtual Future<std::shared_ptr<InspectedFragment>> InspectFragment();

  /// \brief Start a scan operation
  virtual Future<std::shared_ptr<FragmentScanner>> BeginScan(
      const FragmentScanRequest& request, const InspectedFragment& inspected_fragment);

  /// \brief Count the number of rows in this fragment matching the filter using metadata
  /// only. That is, this method may perform I/O, but will not load data.
  ///
  /// If this is not possible, resolve with an empty optional. The fragment can perform
  /// I/O (e.g. to read metadata) before it deciding whether it can satisfy the request.
  virtual Future<util::optional<int64_t>> CountRows(
      compute::Expression predicate, const std::shared_ptr<ScanOptions>& options);

  virtual std::string type_name() const = 0;
  virtual std::string ToString() const { return type_name(); }

  /// \brief An expression which evaluates to true for all data viewed by this
  /// Fragment.
  const compute::Expression& partition_expression() const {
    return partition_expression_;
  }

  virtual ~Fragment() = default;

 protected:
  Fragment() = default;
  explicit Fragment(compute::Expression partition_expression,
                    std::shared_ptr<Schema> physical_schema);

  virtual Result<std::shared_ptr<Schema>> ReadPhysicalSchemaImpl() = 0;

  util::Mutex physical_schema_mutex_;
  compute::Expression partition_expression_ = compute::literal(true);
  std::shared_ptr<Schema> physical_schema_;
};

/// \brief Per-scan options for fragment(s) in a dataset.
///
/// These options are not intrinsic to the format or fragment itself, but do affect
/// the results of a scan. These are options which make sense to change between
/// repeated reads of the same dataset, such as format-specific conversion options
/// (that do not affect the schema).
///
/// \ingroup dataset-scanning
class ARROW_DS_EXPORT FragmentScanOptions {
 public:
  virtual std::string type_name() const = 0;
  virtual std::string ToString() const { return type_name(); }
  virtual ~FragmentScanOptions() = default;
};

/// \defgroup dataset-implementations Concrete implementations
///
/// @{

/// \brief A trivial Fragment that yields ScanTask out of a fixed set of
/// RecordBatch.
class ARROW_DS_EXPORT InMemoryFragment : public Fragment {
 public:
  class Scanner;
  InMemoryFragment(std::shared_ptr<Schema> schema, RecordBatchVector record_batches,
                   compute::Expression = compute::literal(true));
  explicit InMemoryFragment(RecordBatchVector record_batches,
                            compute::Expression = compute::literal(true));

  Result<RecordBatchGenerator> ScanBatchesAsync(
      const std::shared_ptr<ScanOptions>& options) override;
  Future<util::optional<int64_t>> CountRows(
      compute::Expression predicate,
      const std::shared_ptr<ScanOptions>& options) override;

  Future<std::shared_ptr<InspectedFragment>> InspectFragment() override;
  Future<std::shared_ptr<FragmentScanner>> BeginScan(
      const FragmentScanRequest& request,
      const InspectedFragment& inspected_fragment) override;

  std::string type_name() const override { return "in-memory"; }

 protected:
  Result<std::shared_ptr<Schema>> ReadPhysicalSchemaImpl() override;

  RecordBatchVector record_batches_;
};

/// @}

/// \brief Rules for converting the dataset schema to and from fragment schemas
class ARROW_DS_EXPORT FragmentEvolutionStrategy {
 public:
  /// This instance will only be destroyed when all scan operations for the
  /// fragment have completed.
  virtual ~FragmentEvolutionStrategy() = default;
  /// \brief A guarantee that applies to all batches of this fragment
  ///
  /// For example, if a fragment is missing one of the fields in the dataset
  /// schema then a typical evolution strategy is to set that field to null.
  ///
  /// So if the column at index 3 is missing then the guarantee is
  /// FieldRef(3) == null
  ///
  /// Individual field guarantees should be AND'd together and returned
  /// as a single expression.
  virtual Result<compute::Expression> GetGuarantee(
      const std::vector<FieldPath>& dataset_schema_selection) const = 0;

  /// \brief Return a fragment schema selection given a dataset schema selection
  ///
  /// For example, if the user wants fields 2 & 4 of the dataset schema and
  /// in this fragment the field 2 is missing and the field 4 is at index 1 then
  /// this should return {1}
  virtual Result<std::vector<FragmentSelectionColumn>> DevolveSelection(
      const std::vector<FieldPath>& dataset_schema_selection) const = 0;

  /// \brief Return a filter expression bound to the fragment schema given
  ///        a filter expression bound to the dataset schema
  ///
  /// The dataset scan filter will first be simplified by the guarantee returned
  /// by GetGuarantee.  This means an evolution that only handles dropping or casting
  /// fields doesn't need to do anything here except return the given filter.
  ///
  /// On the other hand, an evolution that is doing some kind of aliasing will likely
  /// need to convert field references in the filter to the aliased field references
  /// where appropriate.
  virtual Result<compute::Expression> DevolveFilter(
      const compute::Expression& filter) const = 0;

  /// \brief Convert a batch from the fragment schema to the dataset schema
  ///
  /// Typically this involves casting columns from the data type stored on disk
  /// to the data type of the dataset schema.  For example, this fragment might
  /// have columns stored as int32 and the dataset schema might have int64 for
  /// the column.  In this case we should cast the column from int32 to int64.
  ///
  /// Note: A fragment may perform this cast as the data is read from disk.  In
  /// that case a cast might not be needed.
  virtual Result<compute::ExecBatch> EvolveBatch(
      const std::shared_ptr<RecordBatch>& batch) const = 0;

  /// \brief Return a string description of this strategy
  virtual std::string ToString() const = 0;
};

/// \brief Lookup to create a FragmentEvolutionStrategy for a given fragment
class ARROW_DS_EXPORT DatasetEvolutionStrategy {
 public:
  virtual ~DatasetEvolutionStrategy() = default;
  /// \brief Create a strategy for evolving from the given fragment
  ///        to the schema of the given dataset
  virtual std::unique_ptr<FragmentEvolutionStrategy> GetStrategy(
      const Dataset& dataset, const Fragment& fragment,
      const InspectedFragment& inspected_fragment) = 0;

  /// \brief Return a string description of this strategy
  virtual std::string ToString() const = 0;
};

std::unique_ptr<DatasetEvolutionStrategy> MakeBasicDatasetEvolutionStrategy();

/// \brief A container of zero or more Fragments.
///
/// A Dataset acts as a union of Fragments, e.g. files deeply nested in a
/// directory. A Dataset has a schema to which Fragments must align during a
/// scan operation. This is analogous to Avro's reader and writer schema.
class ARROW_DS_EXPORT Dataset : public std::enable_shared_from_this<Dataset> {
 public:
  /// \brief Begin to build a new Scan operation against this Dataset
  Result<std::shared_ptr<ScannerBuilder>> NewScan();

  /// \brief GetFragments returns an iterator of Fragments given a predicate.
  Result<FragmentIterator> GetFragments(compute::Expression predicate);
  Result<FragmentIterator> GetFragments();

  const std::shared_ptr<Schema>& schema() const { return schema_; }

  /// \brief An expression which evaluates to true for all data viewed by this Dataset.
  /// May be null, which indicates no information is available.
  const compute::Expression& partition_expression() const {
    return partition_expression_;
  }

  /// \brief The name identifying the kind of Dataset
  virtual std::string type_name() const = 0;

  /// \brief Return a copy of this Dataset with a different schema.
  ///
  /// The copy will view the same Fragments. If the new schema is not compatible with the
  /// original dataset's schema then an error will be raised.
  virtual Result<std::shared_ptr<Dataset>> ReplaceSchema(
      std::shared_ptr<Schema> schema) const = 0;

  /// \brief Rules used by this dataset to handle schema evolution
  DatasetEvolutionStrategy* evolution_strategy() { return evolution_strategy_.get(); }

  virtual ~Dataset() = default;

 protected:
  explicit Dataset(std::shared_ptr<Schema> schema) : schema_(std::move(schema)) {}

  Dataset(std::shared_ptr<Schema> schema, compute::Expression partition_expression);

  virtual Result<FragmentIterator> GetFragmentsImpl(compute::Expression predicate) = 0;

  std::shared_ptr<Schema> schema_;
  compute::Expression partition_expression_ = compute::literal(true);
  std::unique_ptr<DatasetEvolutionStrategy> evolution_strategy_ =
      MakeBasicDatasetEvolutionStrategy();
};

/// \addtogroup dataset-implementations
///
/// @{

/// \brief A Source which yields fragments wrapping a stream of record batches.
///
/// The record batches must match the schema provided to the source at construction.
class ARROW_DS_EXPORT InMemoryDataset : public Dataset {
 public:
  class RecordBatchGenerator {
   public:
    virtual ~RecordBatchGenerator() = default;
    virtual RecordBatchIterator Get() const = 0;
  };

  /// Construct a dataset from a schema and a factory of record batch iterators.
  InMemoryDataset(std::shared_ptr<Schema> schema,
                  std::shared_ptr<RecordBatchGenerator> get_batches)
      : Dataset(std::move(schema)), get_batches_(std::move(get_batches)) {}

  /// Convenience constructor taking a fixed list of batches
  InMemoryDataset(std::shared_ptr<Schema> schema, RecordBatchVector batches);

  /// Convenience constructor taking a Table
  explicit InMemoryDataset(std::shared_ptr<Table> table);

  std::string type_name() const override { return "in-memory"; }

  Result<std::shared_ptr<Dataset>> ReplaceSchema(
      std::shared_ptr<Schema> schema) const override;

 protected:
  Result<FragmentIterator> GetFragmentsImpl(compute::Expression predicate) override;

  std::shared_ptr<RecordBatchGenerator> get_batches_;
};

/// \brief A Dataset wrapping child Datasets.
class ARROW_DS_EXPORT UnionDataset : public Dataset {
 public:
  /// \brief Construct a UnionDataset wrapping child Datasets.
  ///
  /// \param[in] schema the schema of the resulting dataset.
  /// \param[in] children one or more child Datasets. Their schemas must be identical to
  /// schema.
  static Result<std::shared_ptr<UnionDataset>> Make(std::shared_ptr<Schema> schema,
                                                    DatasetVector children);

  const DatasetVector& children() const { return children_; }

  std::string type_name() const override { return "union"; }

  Result<std::shared_ptr<Dataset>> ReplaceSchema(
      std::shared_ptr<Schema> schema) const override;

 protected:
  Result<FragmentIterator> GetFragmentsImpl(compute::Expression predicate) override;

  explicit UnionDataset(std::shared_ptr<Schema> schema, DatasetVector children)
      : Dataset(std::move(schema)), children_(std::move(children)) {}

  DatasetVector children_;

  friend class UnionDatasetFactory;
};

/// @}

}  // namespace dataset
}  // namespace arrow
