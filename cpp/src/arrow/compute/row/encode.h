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
#include <memory>
#include <vector>

#include "arrow/array/data.h"
#include "arrow/compute/exec.h"
#include "arrow/compute/exec/util.h"
#include "arrow/compute/light_array.h"
#include "arrow/compute/row/row_internal.h"
#include "arrow/memory_pool.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/util/bit_util.h"

namespace arrow {
namespace compute {

/// Converts between Arrow's typical column representation to a row-based representation
///
/// Data is stored as a single array of rows.  Each row combines data from all columns.
/// The conversion is reversible.
///
/// Row-oriented storage is beneficial when there is a need for random access
/// of individual rows and at the same time all included columns are likely to
/// be accessed together, as in the case of hash table key.
///
/// Does not support nested types
class RowEncoder {
 public:
  void Init(const std::vector<RowTableMetadata>& cols, int row_alignment,
            int string_alignment);

  const RowTableMetadata& row_metadata() { return row_metadata_; }

  /// \brief Prepare to encode a collection of columns
  /// \param start_row The starting row to encode
  /// \param num_rows The number of rows to encode
  /// \param cols The columns to encode.  The order of the columns should
  ///             be consistent with the order used to create the RowTableMetadata
  void PrepareEncodeSelected(int64_t start_row, int64_t num_rows,
                             const std::vector<KeyColumnArray>& cols);
  /// \brief Encode selection of prepared rows into a row table
  /// \param rows The output row table
  /// \param num_selected The number of rows to encode
  /// \param selection indices of the rows to encode
  Status EncodeSelected(RowTable* rows, uint32_t num_selected,
                        const uint16_t* selection);

  /// \brief Decode a window of row oriented data into a corresponding
  ///        window of column oriented storage.
  /// \param start_row_input The starting row to decode
  /// \param start_row_output An offset into the output array to write to
  /// \param num_rows The number of rows to decode
  /// \param rows The row table to decode from
  /// \param cols The columns to decode into, should be sized appropriately
  /// \param hardware_flags Flags to control the decoding
  /// \param temp_stack Preallocated space for temporary allocations
  ///
  /// The output buffers need to be correctly allocated and sized before
  /// calling each method.  For that reason decoding is split into two functions.
  /// DecodeFixedLengthBuffers processes everything except for varying length
  /// buffers.
  /// The output can be used to find out required varying length buffers sizes
  /// for the call to DecodeVaryingLengthBuffers
  void DecodeFixedLengthBuffers(int64_t start_row_input, int64_t start_row_output,
                                int64_t num_rows, const RowTable& rows,
                                std::vector<KeyColumnArray>* cols, int64_t hardware_flags,
                                util::TempVectorStack* temp_stack);

  /// \brief Decode the varlength columns of a row table into column storage
  /// \param start_row_input The starting row to decode
  /// \param start_row_output An offset into the output arrays
  /// \param num_rows The number of rows to decode
  /// \param rows The row table to decode from
  /// \param cols The column arrays to decode into
  /// \param hardware_flags Flags to control the decoding
  /// \param temp_stack Preallocated space for temporary allocations
  void DecodeVaryingLengthBuffers(int64_t start_row_input, int64_t start_row_output,
                                  int64_t num_rows, const RowTable& rows,
                                  std::vector<KeyColumnArray>* cols,
                                  int64_t hardware_flags,
                                  util::TempVectorStack* temp_stack);

 private:
  /// Prepare column array vectors.
  /// Output column arrays represent a range of input column arrays
  /// specified by starting row and number of rows.
  /// Three vectors are generated:
  /// - all columns
  /// - fixed-length columns only
  /// - varying-length columns only
  void PrepareKeyColumnArrays(int64_t start_row, int64_t num_rows,
                              const std::vector<KeyColumnArray>& cols_in);
  
  // Data initialized once, based on data types of key columns
  RowTableMetadata row_metadata_;

  // Data initialized for each input batch.
  // All elements are ordered according to the order of encoded fields in a row.
  std::vector<KeyColumnArray> batch_all_cols_;
  std::vector<KeyColumnArray> batch_varbinary_cols_;
  std::vector<uint32_t> batch_varbinary_cols_base_offsets_;
};

// Write operations (appending batch rows) must not be called by more than one
// thread at the same time.
//
// Read operations (row comparison, column decoding)
// can be called by multiple threads concurrently.
//
struct RowArray {
  RowArray() : is_initialized_(false) {}

  Status InitIfNeeded(MemoryPool* pool, const ExecBatch& batch);
  Status InitIfNeeded(MemoryPool* pool, const RowTableMetadata& row_metadata);

  Status AppendBatchSelection(MemoryPool* pool, const ExecBatch& batch, int begin_row_id,
                              int end_row_id, int num_row_ids, const uint16_t* row_ids,
                              std::vector<KeyColumnArray>& temp_column_arrays);

  // This can only be called for a minibatch.
  //
  void Compare(const ExecBatch& batch, int begin_row_id, int end_row_id, int num_selected,
               const uint16_t* batch_selection_maybe_null, const uint32_t* array_row_ids,
               uint32_t* out_num_not_equal, uint16_t* out_not_equal_selection,
               int64_t hardware_flags, util::TempVectorStack* temp_stack,
               std::vector<KeyColumnArray>& temp_column_arrays,
               uint8_t* out_match_bitvector_maybe_null = NULLPTR);

  // TODO: add AVX2 version
  //
  Status DecodeSelected(ResizableArrayData* target, int column_id, int num_rows_to_append,
                        const uint32_t* row_ids, MemoryPool* pool) const;

  int64_t num_rows() const { return is_initialized_ ? rows_.length() : 0; }

  bool is_initialized_;
  RowEncoder encoder_;
  RowTable rows_;
  RowTable rows_temp_;
};

// Implements concatenating multiple row arrays into a single one, using
// potentially multiple threads, each processing a single input row array.
//
class RowArrayMerge {
 public:
  // Calculate total number of rows and size in bytes for merged sequence of
  // rows and allocate memory for it.
  //
  // If the rows are of varying length, initialize in the offset array the first
  // entry for the write area for each input row array. Leave all other
  // offsets and buffers uninitialized.
  //
  // All input sources must be initialized, but they can contain zero rows.
  //
  // Output in vector the first target row id for each source (exclusive
  // cummulative sum of number of rows in sources).
  //
  static Status PrepareForMerge(RowArray* target, const std::vector<RowArray*>& sources,
                                std::vector<int64_t>* first_target_row_id,
                                MemoryPool* pool);

  // Copy rows from source array to target array.
  // Both arrays must have the same row metadata.
  // Target array must already have the memory reserved in all internal buffers
  // for the copy of the rows.
  //
  // Copy of the rows will occupy the same amount of space in the target array
  // buffers as in the source array, but in the target array we pick at what row
  // position and offset we start writing.
  //
  // Optionally, the rows may be reordered during copy according to the
  // provided permutation, which represents some sorting order of source rows.
  // Nth element of the permutation array is the source row index for the Nth
  // row written into target array. If permutation is missing (null), then the
  // order of source rows will remain unchanged.
  //
  // In case of varying length rows, we purposefully skip outputting of N+1 (one
  // after last) offset, to allow concurrent copies of rows done to adjacent
  // ranges in the target array. This offset should already contain the right
  // value after calling the method preparing target array for merge (which
  // initializes boundary offsets for target row ranges for each source).
  //
  static void MergeSingle(RowArray* target, const RowArray& source,
                          int64_t first_target_row_id,
                          const int64_t* source_rows_permutation);

 private:
  // Copy rows from source array to a region of the target array.
  // This implementation is for fixed length rows.
  // Null information needs to be handled separately.
  //
  static void CopyFixedLength(RowTable* target,
                              const RowTable& source,
                              int64_t first_target_row_id,
                              const int64_t* source_rows_permutation);

  // Copy rows from source array to a region of the target array.
  // This implementation is for varying length rows.
  // Null information needs to be handled separately.
  //
  static void CopyVaryingLength(RowTable* target,
                                const RowTable& source,
                                int64_t first_target_row_id,
                                int64_t first_target_row_offset,
                                const int64_t* source_rows_permutation);

  // Copy null information from rows from source array to a region of the target
  // array.
  //
  static void CopyNulls(RowTable* target,
                        const RowTable& source,
                        int64_t first_target_row_id,
                        const int64_t* source_rows_permutation);
};

/// \brief Helper class for visiting data in a row array
class RowArrayAccessor {
 public:
  // Find the index of this varbinary column within the sequence of all
  // varbinary columns encoded in rows.
  //
  static int VarbinaryColumnId(const RowTableMetadata& row_metadata,
                               int column_id);

  // Calculate how many rows to skip from the tail of the
  // sequence of selected rows, such that the total size of skipped rows is at
  // least equal to the size specified by the caller. Skipping of the tail rows
  // is used to allow for faster processing by the caller of remaining rows
  // without checking buffer bounds (useful with SIMD or fixed size memory loads
  // and stores).
  //
  static int NumRowsToSkip(const RowTable& rows, int column_id,
                           int num_rows, const uint32_t* row_ids,
                           int num_tail_bytes_to_skip);

  // The supplied lambda will be called for each row in the given list of rows.
  // The arguments given to it will be:
  // - index of a row (within the set of selected rows),
  // - pointer to the value,
  // - byte length of the value.
  //
  // The information about nulls (validity bitmap) is not used in this call and
  // has to be processed separately.
  //
  template <class PROCESS_VALUE_FN>
  static void Visit(const RowTable& rows, int column_id, int num_rows,
                    const uint32_t* row_ids, PROCESS_VALUE_FN process_value_fn);

  // The supplied lambda will be called for each row in the given list of rows.
  // The arguments given to it will be:
  // - index of a row (within the set of selected rows),
  // - byte 0xFF if the null is set for the row or 0x00 otherwise.
  //
  template <class PROCESS_VALUE_FN>
  static void VisitNulls(const RowTable& rows, int column_id, int num_rows,
                         const uint32_t* row_ids, PROCESS_VALUE_FN process_value_fn);

 private:
#if defined(ARROW_HAVE_AVX2)
  // This is equivalent to Visit method, but processing 8 rows at a time in a
  // loop.
  // Returns the number of processed rows, which may be less than requested (up
  // to 7 rows at the end may be skipped).
  //
  template <class PROCESS_8_VALUES_FN>
  static int Visit_avx2(const KeyEncoder::RowTable& rows, int column_id, int num_rows,
                        const uint32_t* row_ids, PROCESS_8_VALUES_FN process_8_values_fn);

  // This is equivalent to VisitNulls method, but processing 8 rows at a time in
  // a loop. Returns the number of processed rows, which may be less than
  // requested (up to 7 rows at the end may be skipped).
  //
  template <class PROCESS_8_VALUES_FN>
  static int VisitNulls_avx2(const KeyEncoder::RowTable& rows, int column_id,
                             int num_rows, const uint32_t* row_ids,
                             PROCESS_8_VALUES_FN process_8_values_fn);
#endif
};

}  // namespace compute
}  // namespace arrow
