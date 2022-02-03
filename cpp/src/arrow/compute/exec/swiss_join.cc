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

#include "arrow/compute/exec/swiss_join.h"
#include <sys/stat.h>
#include <algorithm>  // std::upper_bound
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include "arrow/array/util.h"  // MakeArrayFromScalar
#include "arrow/compute/exec/hash_join.h"
#include "arrow/compute/exec/key_compare.h"
#include "arrow/compute/exec/key_encode.h"
#include "arrow/compute/exec/key_hash.h"
#include "arrow/compute/exec/util.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/bitmap_ops.h"

namespace arrow {
namespace compute {

void ResizableArrayData::Init(const std::shared_ptr<DataType>& data_type,
                              MemoryPool* pool, int log_num_rows_min) {
#ifndef NDEBUG
  if (num_rows_allocated_ > 0) {
    ARROW_DCHECK(data_type_ != NULLPTR);
    KeyEncoder::KeyColumnMetadata metadata_before =
        ColumnMetadataFromDataType(data_type_);
    KeyEncoder::KeyColumnMetadata metadata_after = ColumnMetadataFromDataType(data_type);
    ARROW_DCHECK(metadata_before.is_fixed_length == metadata_after.is_fixed_length &&
                 metadata_before.fixed_length == metadata_after.fixed_length);
  }
#endif
  Clear(/*release_buffers=*/false);
  log_num_rows_min_ = log_num_rows_min;
  data_type_ = data_type;
  pool_ = pool;
}

void ResizableArrayData::Clear(bool release_buffers) {
  num_rows_ = 0;
  if (release_buffers) {
    non_null_buf_.reset();
    fixed_len_buf_.reset();
    var_len_buf_.reset();
    num_rows_allocated_ = 0;
    var_len_buf_size_ = 0;
  }
}

Status ResizableArrayData::ResizeFixedLengthBuffers(int num_rows_new) {
  ARROW_DCHECK(num_rows_new >= 0);
  if (num_rows_new <= num_rows_allocated_) {
    num_rows_ = num_rows_new;
    return Status::OK();
  }

  int num_rows_allocated_new = 1 << log_num_rows_min_;
  while (num_rows_allocated_new < num_rows_new) {
    num_rows_allocated_new *= 2;
  }

  KeyEncoder::KeyColumnMetadata column_metadata = ColumnMetadataFromDataType(data_type_);

  if (fixed_len_buf_ == NULLPTR) {
    ARROW_DCHECK(non_null_buf_ == NULLPTR && var_len_buf_ == NULLPTR);

    ARROW_ASSIGN_OR_RAISE(
        non_null_buf_,
        AllocateResizableBuffer(
            bit_util::BytesForBits(num_rows_allocated_new) + kNumPaddingBytes, pool_));
    if (column_metadata.is_fixed_length) {
      if (column_metadata.fixed_length == 0) {
        ARROW_ASSIGN_OR_RAISE(
            fixed_len_buf_,
            AllocateResizableBuffer(
                bit_util::BytesForBits(num_rows_allocated_new) + kNumPaddingBytes,
                pool_));
      } else {
        ARROW_ASSIGN_OR_RAISE(
            fixed_len_buf_,
            AllocateResizableBuffer(
                num_rows_allocated_new * column_metadata.fixed_length + kNumPaddingBytes,
                pool_));
      }
    } else {
      ARROW_ASSIGN_OR_RAISE(
          fixed_len_buf_,
          AllocateResizableBuffer(
              (num_rows_allocated_new + 1) * sizeof(uint32_t) + kNumPaddingBytes, pool_));
    }

    ARROW_ASSIGN_OR_RAISE(var_len_buf_, AllocateResizableBuffer(
                                            sizeof(uint64_t) + kNumPaddingBytes, pool_));

    var_len_buf_size_ = sizeof(uint64_t);
  } else {
    ARROW_DCHECK(non_null_buf_ != NULLPTR && var_len_buf_ != NULLPTR);

    RETURN_NOT_OK(non_null_buf_->Resize(bit_util::BytesForBits(num_rows_allocated_new) +
                                        kNumPaddingBytes));

    if (column_metadata.is_fixed_length) {
      if (column_metadata.fixed_length == 0) {
        RETURN_NOT_OK(fixed_len_buf_->Resize(
            bit_util::BytesForBits(num_rows_allocated_new) + kNumPaddingBytes));
      } else {
        RETURN_NOT_OK(fixed_len_buf_->Resize(
            num_rows_allocated_new * column_metadata.fixed_length + kNumPaddingBytes));
      }
    } else {
      RETURN_NOT_OK(fixed_len_buf_->Resize(
          (num_rows_allocated_new + 1) * sizeof(uint32_t) + kNumPaddingBytes));
    }
  }

  num_rows_allocated_ = num_rows_allocated_new;
  num_rows_ = num_rows_new;

  return Status::OK();
}

Status ResizableArrayData::ResizeVaryingLengthBuffer() {
  KeyEncoder::KeyColumnMetadata column_metadata;
  column_metadata = ColumnMetadataFromDataType(data_type_);

  if (!column_metadata.is_fixed_length) {
    int min_new_size = static_cast<int>(
        reinterpret_cast<const uint32_t*>(fixed_len_buf_->data())[num_rows_]);
    ARROW_DCHECK(var_len_buf_size_ > 0);
    if (var_len_buf_size_ < min_new_size) {
      int new_size = var_len_buf_size_;
      while (new_size < min_new_size) {
        new_size *= 2;
      }
      RETURN_NOT_OK(var_len_buf_->Resize(new_size + kNumPaddingBytes));
      var_len_buf_size_ = new_size;
    }
  }

  return Status::OK();
}

KeyEncoder::KeyColumnArray ResizableArrayData::column_array() const {
  KeyEncoder::KeyColumnMetadata column_metadata;
  column_metadata = ColumnMetadataFromDataType(data_type_);
  return KeyEncoder::KeyColumnArray(
      column_metadata, num_rows_, non_null_buf_->mutable_data(),
      fixed_len_buf_->mutable_data(), var_len_buf_->mutable_data());
}

std::shared_ptr<ArrayData> ResizableArrayData::array_data() const {
  KeyEncoder::KeyColumnMetadata column_metadata;
  column_metadata = ColumnMetadataFromDataType(data_type_);

  auto valid_count = arrow::internal::CountSetBits(non_null_buf_->data(), /*offset=*/0,
                                                   static_cast<int64_t>(num_rows_));
  int null_count = static_cast<int>(num_rows_) - static_cast<int>(valid_count);

  if (column_metadata.is_fixed_length) {
    return ArrayData::Make(data_type_, num_rows_, {non_null_buf_, fixed_len_buf_},
                           null_count);
  } else {
    return ArrayData::Make(data_type_, num_rows_,
                           {non_null_buf_, fixed_len_buf_, var_len_buf_}, null_count);
  }
}

int ExecBatchBuilder::NumRowsToSkip(const std::shared_ptr<ArrayData>& column,
                                    int num_rows, const uint16_t* row_ids,
                                    int num_tail_bytes_to_skip) {
#ifndef NDEBUG
  // Ids must be in non-decreasing order
  //
  for (int i = 1; i < num_rows; ++i) {
    ARROW_DCHECK(row_ids[i] >= row_ids[i - 1]);
  }
#endif

  KeyEncoder::KeyColumnMetadata column_metadata =
      ColumnMetadataFromDataType(column->type);

  int num_rows_left = num_rows;
  int num_bytes_skipped = 0;
  while (num_rows_left > 0 && num_bytes_skipped < num_tail_bytes_to_skip) {
    if (column_metadata.is_fixed_length) {
      if (column_metadata.fixed_length == 0) {
        num_rows_left = std::max(num_rows_left, 8) - 8;
        ++num_bytes_skipped;
      } else {
        --num_rows_left;
        num_bytes_skipped += column_metadata.fixed_length;
      }
    } else {
      --num_rows_left;
      int row_id_removed = row_ids[num_rows_left];
      const uint32_t* offsets =
          reinterpret_cast<const uint32_t*>(column->buffers[1]->data());
      num_bytes_skipped += offsets[row_id_removed + 1] - offsets[row_id_removed];
    }
  }

  return num_rows - num_rows_left;
}

template <bool OUTPUT_BYTE_ALIGNED>
void ExecBatchBuilder::CollectBitsImp(const uint8_t* input_bits,
                                      int64_t input_bits_offset, uint8_t* output_bits,
                                      int64_t output_bits_offset, int num_rows,
                                      const uint16_t* row_ids) {
  if (!OUTPUT_BYTE_ALIGNED) {
    ARROW_DCHECK(output_bits_offset % 8 > 0);
    output_bits[output_bits_offset / 8] &=
        static_cast<uint8_t>((1 << (output_bits_offset % 8)) - 1);
  } else {
    ARROW_DCHECK(output_bits_offset % 8 == 0);
  }
  constexpr int unroll = 8;
  for (int i = 0; i < num_rows / unroll; ++i) {
    const uint16_t* row_ids_base = row_ids + unroll * i;
    uint8_t result;
    result = bit_util::GetBit(input_bits, input_bits_offset + row_ids_base[0]) ? 1 : 0;
    result |= bit_util::GetBit(input_bits, input_bits_offset + row_ids_base[1]) ? 2 : 0;
    result |= bit_util::GetBit(input_bits, input_bits_offset + row_ids_base[2]) ? 4 : 0;
    result |= bit_util::GetBit(input_bits, input_bits_offset + row_ids_base[3]) ? 8 : 0;
    result |= bit_util::GetBit(input_bits, input_bits_offset + row_ids_base[4]) ? 16 : 0;
    result |= bit_util::GetBit(input_bits, input_bits_offset + row_ids_base[5]) ? 32 : 0;
    result |= bit_util::GetBit(input_bits, input_bits_offset + row_ids_base[6]) ? 64 : 0;
    result |= bit_util::GetBit(input_bits, input_bits_offset + row_ids_base[7]) ? 128 : 0;
    if (OUTPUT_BYTE_ALIGNED) {
      output_bits[output_bits_offset / 8 + i] = result;
    } else {
      output_bits[output_bits_offset / 8 + i] |=
          static_cast<uint8_t>(result << (output_bits_offset % 8));
      output_bits[output_bits_offset / 8 + i + 1] =
          static_cast<uint8_t>(result >> (8 - (output_bits_offset % 8)));
    }
  }
  if (num_rows % unroll > 0) {
    for (int i = num_rows - (num_rows % unroll); i < num_rows; ++i) {
      bit_util::SetBitTo(output_bits, output_bits_offset + i,
                         bit_util::GetBit(input_bits, input_bits_offset + row_ids[i]));
    }
  }
}

void ExecBatchBuilder::CollectBits(const uint8_t* input_bits, int64_t input_bits_offset,
                                   uint8_t* output_bits, int64_t output_bits_offset,
                                   int num_rows, const uint16_t* row_ids) {
  if (output_bits_offset % 8 > 0) {
    CollectBitsImp<false>(input_bits, input_bits_offset, output_bits, output_bits_offset,
                          num_rows, row_ids);
  } else {
    CollectBitsImp<true>(input_bits, input_bits_offset, output_bits, output_bits_offset,
                         num_rows, row_ids);
  }
}

template <class PROCESS_VALUE_FN>
void ExecBatchBuilder::Visit(const std::shared_ptr<ArrayData>& column, int num_rows,
                             const uint16_t* row_ids, PROCESS_VALUE_FN process_value_fn) {
  KeyEncoder::KeyColumnMetadata metadata = ColumnMetadataFromDataType(column->type);

  if (!metadata.is_fixed_length) {
    const uint8_t* ptr_base = column->buffers[2]->data();
    const uint32_t* offsets =
        reinterpret_cast<const uint32_t*>(column->buffers[1]->data()) + column->offset;
    for (int i = 0; i < num_rows; ++i) {
      uint16_t row_id = row_ids[i];
      const uint8_t* field_ptr = ptr_base + offsets[row_id];
      uint32_t field_length = offsets[row_id + 1] - offsets[row_id];
      process_value_fn(i, field_ptr, field_length);
    }
  } else {
    ARROW_DCHECK(metadata.fixed_length > 0);
    for (int i = 0; i < num_rows; ++i) {
      uint16_t row_id = row_ids[i];
      const uint8_t* field_ptr =
          column->buffers[1]->data() +
          (column->offset + row_id) * static_cast<int64_t>(metadata.fixed_length);
      process_value_fn(i, field_ptr, metadata.fixed_length);
    }
  }
}

Status ExecBatchBuilder::AppendSelected(const std::shared_ptr<ArrayData>& source,
                                        ResizableArrayData& target,
                                        int num_rows_to_append, const uint16_t* row_ids,
                                        MemoryPool* pool) {
  int num_rows_before = target.num_rows();
  ARROW_DCHECK(num_rows_before >= 0);
  int num_rows_after = num_rows_before + num_rows_to_append;
  if (target.num_rows() == 0) {
    target.Init(source->type, pool, kLogNumRows);
  }
  RETURN_NOT_OK(target.ResizeFixedLengthBuffers(num_rows_after));

  KeyEncoder::KeyColumnMetadata column_metadata =
      ColumnMetadataFromDataType(source->type);

  if (column_metadata.is_fixed_length) {
    // Fixed length column
    //
    uint32_t fixed_length = column_metadata.fixed_length;
    switch (fixed_length) {
      case 0:
        CollectBits(source->buffers[1]->data(), source->offset, target.mutable_data(1),
                    num_rows_before, num_rows_to_append, row_ids);
        break;
      case 1:
        Visit(source, num_rows_to_append, row_ids,
              [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
                target.mutable_data(1)[num_rows_before + i] = *ptr;
              });
        break;
      case 2:
        Visit(source, num_rows_to_append, row_ids,
              [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
                reinterpret_cast<uint16_t*>(target.mutable_data(1))[num_rows_before + i] =
                    *reinterpret_cast<const uint16_t*>(ptr);
              });
        break;
      case 4:
        Visit(source, num_rows_to_append, row_ids,
              [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
                reinterpret_cast<uint32_t*>(target.mutable_data(1))[num_rows_before + i] =
                    *reinterpret_cast<const uint32_t*>(ptr);
              });
        break;
      case 8:
        Visit(source, num_rows_to_append, row_ids,
              [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
                reinterpret_cast<uint64_t*>(target.mutable_data(1))[num_rows_before + i] =
                    *reinterpret_cast<const uint64_t*>(ptr);
              });
        break;
      default: {
        int num_rows_to_process =
            num_rows_to_append -
            NumRowsToSkip(source, num_rows_to_append, row_ids, sizeof(uint64_t));
        Visit(source, num_rows_to_process, row_ids,
              [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
                uint64_t* dst = reinterpret_cast<uint64_t*>(
                    target.mutable_data(1) +
                    static_cast<int64_t>(num_bytes) * (num_rows_before + i));
                const uint64_t* src = reinterpret_cast<const uint64_t*>(ptr);
                for (uint32_t word_id = 0;
                     word_id < bit_util::CeilDiv(num_bytes, sizeof(uint64_t));
                     ++word_id) {
                  util::SafeStore<uint64_t>(dst + word_id, util::SafeLoad(src + word_id));
                }
              });
        if (num_rows_to_append > num_rows_to_process) {
          Visit(source, num_rows_to_append - num_rows_to_process,
                row_ids + num_rows_to_process,
                [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
                  uint64_t* dst = reinterpret_cast<uint64_t*>(
                      target.mutable_data(1) +
                      static_cast<int64_t>(num_bytes) *
                          (num_rows_before + num_rows_to_process + i));
                  const uint64_t* src = reinterpret_cast<const uint64_t*>(ptr);
                  memcpy(dst, src, num_bytes);
                });
        }
      }
    }
  } else {
    // Varying length column
    //

    // Step 1: calculate target offsets
    //
    uint32_t* offsets = reinterpret_cast<uint32_t*>(target.mutable_data(1));
    uint32_t sum = num_rows_before == 0 ? 0 : offsets[num_rows_before];
    Visit(source, num_rows_to_append, row_ids,
          [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
            offsets[num_rows_before + i] = num_bytes;
          });
    for (int i = 0; i < num_rows_to_append; ++i) {
      uint32_t length = offsets[num_rows_before + i];
      offsets[num_rows_before + i] = sum;
      sum += length;
    }
    offsets[num_rows_before + num_rows_to_append] = sum;

    // Step 2: resize output buffers
    //
    RETURN_NOT_OK(target.ResizeVaryingLengthBuffer());

    // Step 3: copy varying-length data
    //
    int num_rows_to_process =
        num_rows_to_append -
        NumRowsToSkip(source, num_rows_to_append, row_ids, sizeof(uint64_t));
    Visit(source, num_rows_to_process, row_ids,
          [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
            uint64_t* dst = reinterpret_cast<uint64_t*>(target.mutable_data(2) +
                                                        offsets[num_rows_before + i]);
            const uint64_t* src = reinterpret_cast<const uint64_t*>(ptr);
            for (uint32_t word_id = 0;
                 word_id < bit_util::CeilDiv(num_bytes, sizeof(uint64_t)); ++word_id) {
              util::SafeStore<uint64_t>(dst + word_id, util::SafeLoad(src + word_id));
            }
          });
    Visit(source, num_rows_to_append - num_rows_to_process, row_ids + num_rows_to_process,
          [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
            uint64_t* dst = reinterpret_cast<uint64_t*>(
                target.mutable_data(2) +
                offsets[num_rows_before + num_rows_to_process + i]);
            const uint64_t* src = reinterpret_cast<const uint64_t*>(ptr);
            memcpy(dst, src, num_bytes);
          });
  }

  // Process nulls
  //
  if (source->buffers[0] == NULLPTR) {
    uint8_t* dst = target.mutable_data(0);
    dst[num_rows_before / 8] |= static_cast<uint8_t>(~0ULL << (num_rows_before & 7));
    for (int i = num_rows_before / 8 + 1;
         i < bit_util::BytesForBits(num_rows_before + num_rows_to_append); ++i) {
      dst[i] = 0xff;
    }
  } else {
    CollectBits(source->buffers[0]->data(), source->offset, target.mutable_data(0),
                num_rows_before, num_rows_to_append, row_ids);
  }

  return Status::OK();
}

Status ExecBatchBuilder::AppendNulls(const std::shared_ptr<DataType>& type,
                                     ResizableArrayData& target, int num_rows_to_append,
                                     MemoryPool* pool) {
  int num_rows_before = target.num_rows();
  int num_rows_after = num_rows_before + num_rows_to_append;
  if (target.num_rows() == 0) {
    target.Init(type, pool, kLogNumRows);
  }
  RETURN_NOT_OK(target.ResizeFixedLengthBuffers(num_rows_after));

  KeyEncoder::KeyColumnMetadata column_metadata = ColumnMetadataFromDataType(type);

  // Process fixed length buffer
  //
  if (column_metadata.is_fixed_length) {
    uint8_t* dst = target.mutable_data(1);
    if (column_metadata.fixed_length == 0) {
      dst[num_rows_before / 8] &= static_cast<uint8_t>((1 << (num_rows_before % 8)) - 1);
      int64_t offset_begin = num_rows_before / 8 + 1;
      int64_t offset_end = bit_util::BytesForBits(num_rows_after);
      if (offset_end > offset_begin) {
        memset(dst + offset_begin, 0, offset_end - offset_begin);
      }
    } else {
      memset(dst + num_rows_before * static_cast<int64_t>(column_metadata.fixed_length),
             0, static_cast<int64_t>(column_metadata.fixed_length) * num_rows_to_append);
    }
  } else {
    uint32_t* dst = reinterpret_cast<uint32_t*>(target.mutable_data(1));
    uint32_t sum = num_rows_before == 0 ? 0 : dst[num_rows_before];
    for (int64_t i = num_rows_before; i <= num_rows_after; ++i) {
      dst[i] = sum;
    }
  }

  // Process nulls
  //
  uint8_t* dst = target.mutable_data(0);
  dst[num_rows_before / 8] &= static_cast<uint8_t>((1 << (num_rows_before % 8)) - 1);
  int64_t offset_begin = num_rows_before / 8 + 1;
  int64_t offset_end = bit_util::BytesForBits(num_rows_after);
  if (offset_end > offset_begin) {
    memset(dst + offset_begin, 0, offset_end - offset_begin);
  }

  return Status::OK();
}

Status ExecBatchBuilder::AppendSelected(MemoryPool* pool, const ExecBatch& batch,
                                        int num_rows_to_append, const uint16_t* row_ids,
                                        int num_cols, const int* col_ids) {
  if (num_rows_to_append == 0) {
    return Status::OK();
  }
  // If this is the first time we append rows, then initialize output buffers.
  //
  if (values_.empty()) {
    values_.resize(num_cols);
    for (int i = 0; i < num_cols; ++i) {
      const Datum& data = batch.values[col_ids ? col_ids[i] : i];
      ARROW_DCHECK(data.is_array());
      const std::shared_ptr<ArrayData>& array_data = data.array();
      values_[i].Init(array_data->type, pool, kLogNumRows);
    }
  }

  for (size_t i = 0; i < values_.size(); ++i) {
    const Datum& data = batch.values[col_ids ? col_ids[i] : i];
    ARROW_DCHECK(data.is_array());
    const std::shared_ptr<ArrayData>& array_data = data.array();
    RETURN_NOT_OK(
        AppendSelected(array_data, values_[i], num_rows_to_append, row_ids, pool));
  }

  return Status::OK();
}

Status ExecBatchBuilder::AppendSelected(MemoryPool* pool, const ExecBatch& batch,
                                        int num_rows_to_append, const uint16_t* row_ids,
                                        int* num_appended, int num_cols,
                                        const int* col_ids) {
  *num_appended = 0;
  if (num_rows_to_append == 0) {
    return Status::OK();
  }
  int num_rows_max = 1 << kLogNumRows;
  int num_rows_present = num_rows();
  if (num_rows_present >= num_rows_max) {
    return Status::OK();
  }
  int num_rows_available = num_rows_max - num_rows_present;
  int num_rows_next = std::min(num_rows_available, num_rows_to_append);
  RETURN_NOT_OK(AppendSelected(pool, batch, num_rows_next, row_ids, num_cols, col_ids));
  *num_appended = num_rows_next;
  return Status::OK();
}

Status ExecBatchBuilder::AppendNulls(MemoryPool* pool,
                                     const std::vector<std::shared_ptr<DataType>>& types,
                                     int num_rows_to_append) {
  if (num_rows_to_append == 0) {
    return Status::OK();
  }

  // If this is the first time we append rows, then initialize output buffers.
  //
  if (values_.empty()) {
    values_.resize(types.size());
    for (size_t i = 0; i < types.size(); ++i) {
      values_[i].Init(types[i], pool, kLogNumRows);
    }
  }

  for (size_t i = 0; i < values_.size(); ++i) {
    RETURN_NOT_OK(AppendNulls(types[i], values_[i], num_rows_to_append, pool));
  }

  return Status::OK();
}

Status ExecBatchBuilder::AppendNulls(MemoryPool* pool,
                                     const std::vector<std::shared_ptr<DataType>>& types,
                                     int num_rows_to_append, int* num_appended) {
  *num_appended = 0;
  if (num_rows_to_append == 0) {
    return Status::OK();
  }
  int num_rows_max = 1 << kLogNumRows;
  int num_rows_present = num_rows();
  if (num_rows_present >= num_rows_max) {
    return Status::OK();
  }
  int num_rows_available = num_rows_max - num_rows_present;
  int num_rows_next = std::min(num_rows_available, num_rows_to_append);
  RETURN_NOT_OK(AppendNulls(pool, types, num_rows_next));
  *num_appended = num_rows_next;
  return Status::OK();
}

ExecBatch ExecBatchBuilder::Flush() {
  ARROW_DCHECK(num_rows() > 0);
  ExecBatch out({}, num_rows());
  out.values.resize(values_.size());
  for (size_t i = 0; i < values_.size(); ++i) {
    out.values[i] = values_[i].array_data();
    values_[i].Clear(true);
  }
  return out;
}

}  // namespace compute
}  // namespace arrow
