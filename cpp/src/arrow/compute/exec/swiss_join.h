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
#include "arrow/compute/exec/key_encode.h"
#include "arrow/compute/exec/key_map.h"
#include "arrow/compute/exec/options.h"
#include "arrow/compute/exec/partition_util.h"
#include "arrow/compute/exec/schema_util.h"
#include "arrow/compute/exec/task_util.h"

namespace arrow {
namespace compute {

class ResizableArrayData {
 public:
  ResizableArrayData()
      : log_num_rows_min_(0),
        pool_(NULLPTR),
        num_rows_(0),
        num_rows_allocated_(0),
        var_len_buf_size_(0) {}
  ~ResizableArrayData() { Clear(true); }
  void Init(const std::shared_ptr<DataType>& data_type, MemoryPool* pool,
            int log_num_rows_min);
  void Clear(bool release_buffers);
  Status ResizeFixedLengthBuffers(int num_rows_new);
  Status ResizeVaryingLengthBuffer();
  int num_rows() const { return num_rows_; }
  KeyEncoder::KeyColumnArray column_array() const;
  KeyEncoder::KeyColumnMetadata column_metadata() const {
    return ColumnMetadataFromDataType(data_type_);
  }
  std::shared_ptr<ArrayData> array_data() const;
  uint8_t* mutable_data(int i) {
    return i == 0   ? non_null_buf_->mutable_data()
           : i == 1 ? fixed_len_buf_->mutable_data()
                    : var_len_buf_->mutable_data();
  }

 private:
  static constexpr int64_t kNumPaddingBytes = 64;
  int log_num_rows_min_;
  std::shared_ptr<DataType> data_type_;
  MemoryPool* pool_;
  int num_rows_;
  int num_rows_allocated_;
  int var_len_buf_size_;
  std::shared_ptr<ResizableBuffer> non_null_buf_;
  std::shared_ptr<ResizableBuffer> fixed_len_buf_;
  std::shared_ptr<ResizableBuffer> var_len_buf_;
};

class ExecBatchBuilder {
 public:
  static Status AppendSelected(const std::shared_ptr<ArrayData>& source,
                               ResizableArrayData& target, int num_rows_to_append,
                               const uint16_t* row_ids, MemoryPool* pool);

  static Status AppendNulls(const std::shared_ptr<DataType>& type,
                            ResizableArrayData& target, int num_rows_to_append,
                            MemoryPool* pool);

  Status AppendSelected(MemoryPool* pool, const ExecBatch& batch, int num_rows_to_append,
                        const uint16_t* row_ids, int num_cols,
                        const int* col_ids = NULLPTR);

  Status AppendSelected(MemoryPool* pool, const ExecBatch& batch, int num_rows_to_append,
                        const uint16_t* row_ids, int* num_appended, int num_cols,
                        const int* col_ids = NULLPTR);

  Status AppendNulls(MemoryPool* pool,
                     const std::vector<std::shared_ptr<DataType>>& types,
                     int num_rows_to_append);

  Status AppendNulls(MemoryPool* pool,
                     const std::vector<std::shared_ptr<DataType>>& types,
                     int num_rows_to_append, int* num_appended);

  // Should only be called if num_rows() returns non-zero.
  //
  ExecBatch Flush();

  int num_rows() const { return values_.empty() ? 0 : values_[0].num_rows(); }

  static int num_rows_max() { return 1 << kLogNumRows; }

 private:
  static constexpr int kLogNumRows = 15;

  // Calculate how many rows to skip from the tail of the
  // sequence of selected rows, such that the total size of skipped rows is at
  // least equal to the size specified by the caller. Skipping of the tail rows
  // is used to allow for faster processing by the caller of remaining rows
  // without checking buffer bounds (useful with SIMD or fixed size memory loads
  // and stores).
  //
  // The sequence of row_ids provided must be non-decreasing.
  //
  static int NumRowsToSkip(const std::shared_ptr<ArrayData>& column, int num_rows,
                           const uint16_t* row_ids, int num_tail_bytes_to_skip);

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
  static void Visit(const std::shared_ptr<ArrayData>& column, int num_rows,
                    const uint16_t* row_ids, PROCESS_VALUE_FN process_value_fn);

  template <bool OUTPUT_BYTE_ALIGNED>
  static void CollectBitsImp(const uint8_t* input_bits, int64_t input_bits_offset,
                             uint8_t* output_bits, int64_t output_bits_offset,
                             int num_rows, const uint16_t* row_ids);
  static void CollectBits(const uint8_t* input_bits, int64_t input_bits_offset,
                          uint8_t* output_bits, int64_t output_bits_offset, int num_rows,
                          const uint16_t* row_ids);

  std::vector<ResizableArrayData> values_;
};

}  // namespace compute
}  // namespace arrow
