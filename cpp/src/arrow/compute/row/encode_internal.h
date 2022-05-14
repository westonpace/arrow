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
class KeyEncoder {
 public:
  void Init(const std::vector<KeyColumnMetadata>& cols, int row_alignment,
            int string_alignment);

  const KeyRowMetadata& row_metadata() { return row_metadata_; }

  void PrepareEncodeSelected(int64_t start_row, int64_t num_rows,
                             const std::vector<KeyColumnArray>& cols);
  Status EncodeSelected(KeyRowArray* rows, uint32_t num_selected,
                        const uint16_t* selection);

  /// Decode a window of row oriented data into a corresponding
  /// window of column oriented storage.
  /// The output buffers need to be correctly allocated and sized before
  /// calling each method.
  /// For that reason decoding is split into two functions.
  /// The output of the first one, that processes everything except for
  /// varying length buffers, can be used to find out required varying
  /// length buffers sizes.
  void DecodeFixedLengthBuffers(int64_t start_row_input, int64_t start_row_output,
                                int64_t num_rows, const KeyRowArray& rows,
                                std::vector<KeyColumnArray>* cols, int64_t hardware_flags,
                                util::TempVectorStack* temp_stack);

  void DecodeVaryingLengthBuffers(int64_t start_row_input, int64_t start_row_output,
                                  int64_t num_rows, const KeyRowArray& rows,
                                  std::vector<KeyColumnArray>* cols,
                                  int64_t hardware_flags,
                                  util::TempVectorStack* temp_stack);

  const std::vector<KeyColumnArray>& GetBatchColumns() const { return batch_all_cols_; }

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

  class TransformBoolean {
   public:
    static KeyColumnArray ArrayReplace(const KeyColumnArray& column,
                                       const KeyColumnArray& temp);
    static void PostDecode(const KeyColumnArray& input, KeyColumnArray* output,
                           KeyEncoderContext* ctx);
  };

  class EncoderInteger {
   public:
    static void Decode(uint32_t start_row, uint32_t num_rows, uint32_t offset_within_row,
                       const KeyRowArray& rows, KeyColumnArray* col,
                       KeyEncoderContext* ctx, KeyColumnArray* temp);
    static bool UsesTransform(const KeyColumnArray& column);
    static KeyColumnArray ArrayReplace(const KeyColumnArray& column,
                                       const KeyColumnArray& temp);
    static void PostDecode(const KeyColumnArray& input, KeyColumnArray* output,
                           KeyEncoderContext* ctx);

   private:
    static bool IsBoolean(const KeyColumnMetadata& metadata);
  };

  class EncoderBinary {
   public:
    static void EncodeSelected(uint32_t offset_within_row, KeyRowArray* rows,
                               const KeyColumnArray& col, uint32_t num_selected,
                               const uint16_t* selection);
    static void Decode(uint32_t start_row, uint32_t num_rows, uint32_t offset_within_row,
                       const KeyRowArray& rows, KeyColumnArray* col,
                       KeyEncoderContext* ctx, KeyColumnArray* temp);
    static bool IsInteger(const KeyColumnMetadata& metadata);

   private:
    template <class COPY_FN, class SET_NULL_FN>
    static void EncodeSelectedImp(uint32_t offset_within_row, KeyRowArray* rows,
                                  const KeyColumnArray& col, uint32_t num_selected,
                                  const uint16_t* selection, COPY_FN copy_fn,
                                  SET_NULL_FN set_null_fn);

    template <bool is_row_fixed_length, class COPY_FN>
    static inline void DecodeHelper(uint32_t start_row, uint32_t num_rows,
                                    uint32_t offset_within_row,
                                    const KeyRowArray* rows_const,
                                    KeyRowArray* rows_mutable_maybe_null,
                                    const KeyColumnArray* col_const,
                                    KeyColumnArray* col_mutable_maybe_null,
                                    COPY_FN copy_fn);
    template <bool is_row_fixed_length>
    static void DecodeImp(uint32_t start_row, uint32_t num_rows,
                          uint32_t offset_within_row, const KeyRowArray& rows,
                          KeyColumnArray* col);
#if defined(ARROW_HAVE_AVX2)
    static void DecodeHelper_avx2(bool is_row_fixed_length, uint32_t start_row,
                                  uint32_t num_rows, uint32_t offset_within_row,
                                  const KeyRowArray& rows, KeyColumnArray* col);
    template <bool is_row_fixed_length>
    static void DecodeImp_avx2(uint32_t start_row, uint32_t num_rows,
                               uint32_t offset_within_row, const KeyRowArray& rows,
                               KeyColumnArray* col);
#endif
  };

  class EncoderBinaryPair {
   public:
    static bool CanProcessPair(const KeyColumnMetadata& col1,
                               const KeyColumnMetadata& col2) {
      return EncoderBinary::IsInteger(col1) && EncoderBinary::IsInteger(col2);
    }
    static void Decode(uint32_t start_row, uint32_t num_rows, uint32_t offset_within_row,
                       const KeyRowArray& rows, KeyColumnArray* col1,
                       KeyColumnArray* col2, KeyEncoderContext* ctx,
                       KeyColumnArray* temp1, KeyColumnArray* temp2);

   private:
    template <bool is_row_fixed_length, typename col1_type, typename col2_type>
    static void DecodeImp(uint32_t num_rows_to_skip, uint32_t start_row,
                          uint32_t num_rows, uint32_t offset_within_row,
                          const KeyRowArray& rows, KeyColumnArray* col1,
                          KeyColumnArray* col2);
#if defined(ARROW_HAVE_AVX2)
    static uint32_t DecodeHelper_avx2(bool is_row_fixed_length, uint32_t col_width,
                                      uint32_t start_row, uint32_t num_rows,
                                      uint32_t offset_within_row, const KeyRowArray& rows,
                                      KeyColumnArray* col1, KeyColumnArray* col2);
    template <bool is_row_fixed_length, uint32_t col_width>
    static uint32_t DecodeImp_avx2(uint32_t start_row, uint32_t num_rows,
                                   uint32_t offset_within_row, const KeyRowArray& rows,
                                   KeyColumnArray* col1, KeyColumnArray* col2);
#endif
  };

  class EncoderOffsets {
   public:
    static void GetRowOffsetsSelected(KeyRowArray* rows,
                                      const std::vector<KeyColumnArray>& cols,
                                      uint32_t num_selected, const uint16_t* selection);
    static void EncodeSelected(KeyRowArray* rows, const std::vector<KeyColumnArray>& cols,
                               uint32_t num_selected, const uint16_t* selection);

    static void Decode(uint32_t start_row, uint32_t num_rows, const KeyRowArray& rows,
                       std::vector<KeyColumnArray>* varbinary_cols,
                       const std::vector<uint32_t>& varbinary_cols_base_offset,
                       KeyEncoderContext* ctx);

   private:
    template <bool has_nulls, bool is_first_varbinary>
    static void EncodeSelectedImp(uint32_t ivarbinary, KeyRowArray* rows,
                                  const std::vector<KeyColumnArray>& cols,
                                  uint32_t num_selected, const uint16_t* selection);
  };

  class EncoderVarBinary {
   public:
    static void EncodeSelected(uint32_t ivarbinary, KeyRowArray* rows,
                               const KeyColumnArray& cols, uint32_t num_selected,
                               const uint16_t* selection);

    static void Decode(uint32_t start_row, uint32_t num_rows, uint32_t varbinary_col_id,
                       const KeyRowArray& rows, KeyColumnArray* col,
                       KeyEncoderContext* ctx);

   private:
    template <bool first_varbinary_col, class COPY_FN>
    static inline void DecodeHelper(uint32_t start_row, uint32_t num_rows,
                                    uint32_t varbinary_col_id,
                                    const KeyRowArray* rows_const,
                                    KeyRowArray* rows_mutable_maybe_null,
                                    const KeyColumnArray* col_const,
                                    KeyColumnArray* col_mutable_maybe_null,
                                    COPY_FN copy_fn);
    template <bool first_varbinary_col>
    static void DecodeImp(uint32_t start_row, uint32_t num_rows,
                          uint32_t varbinary_col_id, const KeyRowArray& rows,
                          KeyColumnArray* col);
#if defined(ARROW_HAVE_AVX2)
    static void DecodeHelper_avx2(uint32_t start_row, uint32_t num_rows,
                                  uint32_t varbinary_col_id, const KeyRowArray& rows,
                                  KeyColumnArray* col);
    template <bool first_varbinary_col>
    static void DecodeImp_avx2(uint32_t start_row, uint32_t num_rows,
                               uint32_t varbinary_col_id, const KeyRowArray& rows,
                               KeyColumnArray* col);
#endif
  };

  class EncoderNulls {
   public:
    static void EncodeSelected(KeyRowArray* rows, const std::vector<KeyColumnArray>& cols,
                               uint32_t num_selected, const uint16_t* selection);

    static void Decode(uint32_t start_row, uint32_t num_rows, const KeyRowArray& rows,
                       std::vector<KeyColumnArray>* cols);
  };

  // Data initialized once, based on data types of key columns
  KeyRowMetadata row_metadata_;

  // Data initialized for each input batch.
  // All elements are ordered according to the order of encoded fields in a row.
  std::vector<KeyColumnArray> batch_all_cols_;
  std::vector<KeyColumnArray> batch_varbinary_cols_;
  std::vector<uint32_t> batch_varbinary_cols_base_offsets_;
};

template <bool is_row_fixed_length, class COPY_FN>
inline void KeyEncoder::EncoderBinary::DecodeHelper(
    uint32_t start_row, uint32_t num_rows, uint32_t offset_within_row,
    const KeyRowArray* rows_const, KeyRowArray* rows_mutable_maybe_null,
    const KeyColumnArray* col_const, KeyColumnArray* col_mutable_maybe_null,
    COPY_FN copy_fn) {
  ARROW_DCHECK(col_const && col_const->metadata().is_fixed_length);
  uint32_t col_width = col_const->metadata().fixed_length;

  if (is_row_fixed_length) {
    uint32_t row_width = rows_const->metadata().fixed_length;
    for (uint32_t i = 0; i < num_rows; ++i) {
      const uint8_t* src;
      uint8_t* dst;
      src = rows_const->data(1) + row_width * (start_row + i) + offset_within_row;
      dst = col_mutable_maybe_null->mutable_data(1) + col_width * i;
      copy_fn(dst, src, col_width);
    }
  } else {
    const uint32_t* row_offsets = rows_const->offsets();
    for (uint32_t i = 0; i < num_rows; ++i) {
      const uint8_t* src;
      uint8_t* dst;
      src = rows_const->data(2) + row_offsets[start_row + i] + offset_within_row;
      dst = col_mutable_maybe_null->mutable_data(1) + col_width * i;
      copy_fn(dst, src, col_width);
    }
  }
}

template <bool first_varbinary_col, class COPY_FN>
inline void KeyEncoder::EncoderVarBinary::DecodeHelper(
    uint32_t start_row, uint32_t num_rows, uint32_t varbinary_col_id,
    const KeyRowArray* rows_const, KeyRowArray* rows_mutable_maybe_null,
    const KeyColumnArray* col_const, KeyColumnArray* col_mutable_maybe_null,
    COPY_FN copy_fn) {
  // Column and rows need to be varying length
  ARROW_DCHECK(!rows_const->metadata().is_fixed_length &&
               !col_const->metadata().is_fixed_length);

  const uint32_t* row_offsets_for_batch = rows_const->offsets() + start_row;
  const uint32_t* col_offsets = col_const->offsets();

  uint32_t col_offset_next = col_offsets[0];
  for (uint32_t i = 0; i < num_rows; ++i) {
    uint32_t col_offset = col_offset_next;
    col_offset_next = col_offsets[i + 1];

    uint32_t row_offset = row_offsets_for_batch[i];
    const uint8_t* row = rows_const->data(2) + row_offset;

    uint32_t offset_within_row;
    uint32_t length;
    if (first_varbinary_col) {
      rows_const->metadata().first_varbinary_offset_and_length(row, &offset_within_row,
                                                               &length);
    } else {
      rows_const->metadata().nth_varbinary_offset_and_length(row, varbinary_col_id,
                                                             &offset_within_row, &length);
    }

    row_offset += offset_within_row;

    const uint8_t* src;
    uint8_t* dst;
    src = rows_const->data(2) + row_offset;
    dst = col_mutable_maybe_null->mutable_data(2) + col_offset;
    copy_fn(dst, src, length);
  }
}

// Write operations (appending batch rows) must not be called by more than one
// thread at the same time.
//
// Read operations (row comparison, column decoding)
// can be called by multiple threads concurrently.
//
struct RowArray {
  RowArray() : is_initialized_(false) {}

  Status InitIfNeeded(MemoryPool* pool, const ExecBatch& batch);
  Status InitIfNeeded(MemoryPool* pool, const KeyEncoder::KeyRowMetadata& row_metadata);

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
  KeyEncoder encoder_;
  KeyRowArray rows_;
  KeyRowArray rows_temp_;
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
  static void CopyFixedLength(KeyEncoder::KeyRowArray* target,
                              const KeyEncoder::KeyRowArray& source,
                              int64_t first_target_row_id,
                              const int64_t* source_rows_permutation);

  // Copy rows from source array to a region of the target array.
  // This implementation is for varying length rows.
  // Null information needs to be handled separately.
  //
  static void CopyVaryingLength(KeyEncoder::KeyRowArray* target,
                                const KeyEncoder::KeyRowArray& source,
                                int64_t first_target_row_id,
                                int64_t first_target_row_offset,
                                const int64_t* source_rows_permutation);

  // Copy null information from rows from source array to a region of the target
  // array.
  //
  static void CopyNulls(KeyEncoder::KeyRowArray* target,
                        const KeyEncoder::KeyRowArray& source,
                        int64_t first_target_row_id,
                        const int64_t* source_rows_permutation);
};

/// \brief Helper class for visiting data in a row array
class RowArrayAccessor {
 public:
  // Find the index of this varbinary column within the sequence of all
  // varbinary columns encoded in rows.
  //
  static int VarbinaryColumnId(const KeyEncoder::KeyRowMetadata& row_metadata,
                               int column_id);

  // Calculate how many rows to skip from the tail of the
  // sequence of selected rows, such that the total size of skipped rows is at
  // least equal to the size specified by the caller. Skipping of the tail rows
  // is used to allow for faster processing by the caller of remaining rows
  // without checking buffer bounds (useful with SIMD or fixed size memory loads
  // and stores).
  //
  static int NumRowsToSkip(const KeyEncoder::KeyRowArray& rows, int column_id,
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
  static void Visit(const KeyEncoder::KeyRowArray& rows, int column_id, int num_rows,
                    const uint32_t* row_ids, PROCESS_VALUE_FN process_value_fn);

  // The supplied lambda will be called for each row in the given list of rows.
  // The arguments given to it will be:
  // - index of a row (within the set of selected rows),
  // - byte 0xFF if the null is set for the row or 0x00 otherwise.
  //
  template <class PROCESS_VALUE_FN>
  static void VisitNulls(const KeyEncoder::KeyRowArray& rows, int column_id, int num_rows,
                         const uint32_t* row_ids, PROCESS_VALUE_FN process_value_fn);

 private:
#if defined(ARROW_HAVE_AVX2)
  // This is equivalent to Visit method, but processing 8 rows at a time in a
  // loop.
  // Returns the number of processed rows, which may be less than requested (up
  // to 7 rows at the end may be skipped).
  //
  template <class PROCESS_8_VALUES_FN>
  static int Visit_avx2(const KeyEncoder::KeyRowArray& rows, int column_id, int num_rows,
                        const uint32_t* row_ids, PROCESS_8_VALUES_FN process_8_values_fn);

  // This is equivalent to VisitNulls method, but processing 8 rows at a time in
  // a loop. Returns the number of processed rows, which may be less than
  // requested (up to 7 rows at the end may be skipped).
  //
  template <class PROCESS_8_VALUES_FN>
  static int VisitNulls_avx2(const KeyEncoder::KeyRowArray& rows, int column_id,
                             int num_rows, const uint32_t* row_ids,
                             PROCESS_8_VALUES_FN process_8_values_fn);
#endif
};

}  // namespace compute
}  // namespace arrow
