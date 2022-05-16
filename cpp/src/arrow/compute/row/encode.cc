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

#include "arrow/compute/row/key_encode_internal.h"
#include "arrow/compute/row/row_internal.h"

#include <memory.h>

#include <algorithm>

#include "arrow/compute/exec.h"
#include "arrow/compute/exec/util.h"
#include "arrow/util/bit_util.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/ubsan.h"

namespace arrow {
namespace compute {

void RowEncoder::Init(const std::vector<KeyColumnMetadata>& cols, int row_alignment,
                      int string_alignment) {
  row_metadata_.FromColumnMetadataVector(cols, row_alignment, string_alignment);
  uint32_t num_cols = row_metadata_.num_cols();
  uint32_t num_varbinary_cols = row_metadata_.num_varbinary_cols();
  batch_all_cols_.resize(num_cols);
  batch_varbinary_cols_.resize(num_varbinary_cols);
  batch_varbinary_cols_base_offsets_.resize(num_varbinary_cols);
}

void RowEncoder::PrepareKeyColumnArrays(int64_t start_row, int64_t num_rows,
                                        const std::vector<KeyColumnArray>& cols_in) {
  const auto num_cols = static_cast<uint32_t>(cols_in.size());
  DCHECK(batch_all_cols_.size() == num_cols);

  uint32_t num_varbinary_visited = 0;
  for (uint32_t i = 0; i < num_cols; ++i) {
    const KeyColumnArray& col = cols_in[row_metadata_.column_order[i]];
    KeyColumnArray col_window = col.Slice(start_row, num_rows);

    batch_all_cols_[i] = col_window;
    if (!col.metadata().is_fixed_length) {
      DCHECK(num_varbinary_visited < batch_varbinary_cols_.size());
      // If start row is zero, then base offset of varbinary column is also zero.
      if (start_row == 0) {
        batch_varbinary_cols_base_offsets_[num_varbinary_visited] = 0;
      } else {
        batch_varbinary_cols_base_offsets_[num_varbinary_visited] =
            col.offsets()[start_row];
      }
      batch_varbinary_cols_[num_varbinary_visited++] = col_window;
    }
  }
}

void RowEncoder::DecodeFixedLengthBuffers(int64_t start_row_input,
                                          int64_t start_row_output, int64_t num_rows,
                                          const RowTable& rows,
                                          std::vector<KeyColumnArray>* cols,
                                          int64_t hardware_flags,
                                          util::TempVectorStack* temp_stack) {
  // Prepare column array vectors
  PrepareKeyColumnArrays(start_row_output, num_rows, *cols);

  RowEncoderContext ctx;
  ctx.hardware_flags = hardware_flags;
  ctx.stack = temp_stack;

  // Create two temp vectors with 16-bit elements
  auto temp_buffer_holder_A =
      util::TempVectorHolder<uint16_t>(ctx.stack, static_cast<uint32_t>(num_rows));
  auto temp_buffer_A = KeyColumnArray(
      KeyColumnMetadata(true, sizeof(uint16_t)), num_rows, nullptr,
      reinterpret_cast<uint8_t*>(temp_buffer_holder_A.mutable_data()), nullptr);
  auto temp_buffer_holder_B =
      util::TempVectorHolder<uint16_t>(ctx.stack, static_cast<uint32_t>(num_rows));
  auto temp_buffer_B = KeyColumnArray(
      KeyColumnMetadata(true, sizeof(uint16_t)), num_rows, nullptr,
      reinterpret_cast<uint8_t*>(temp_buffer_holder_B.mutable_data()), nullptr);

  bool is_row_fixed_length = row_metadata_.is_fixed_length;
  if (!is_row_fixed_length) {
    EncoderOffsets::Decode(static_cast<uint32_t>(start_row_input),
                           static_cast<uint32_t>(num_rows), rows, &batch_varbinary_cols_,
                           batch_varbinary_cols_base_offsets_, &ctx);
  }

  // Process fixed length columns
  const auto num_cols = static_cast<uint32_t>(batch_all_cols_.size());
  for (uint32_t i = 0; i < num_cols;) {
    if (!batch_all_cols_[i].metadata().is_fixed_length ||
        batch_all_cols_[i].metadata().is_null_type) {
      i += 1;
      continue;
    }
    bool can_process_pair =
        (i + 1 < num_cols) && batch_all_cols_[i + 1].metadata().is_fixed_length &&
        EncoderBinaryPair::CanProcessPair(batch_all_cols_[i].metadata(),
                                          batch_all_cols_[i + 1].metadata());
    if (!can_process_pair) {
      EncoderBinary::Decode(static_cast<uint32_t>(start_row_input),
                            static_cast<uint32_t>(num_rows),
                            row_metadata_.column_offsets[i], rows, &batch_all_cols_[i],
                            &ctx, &temp_buffer_A);
      i += 1;
    } else {
      EncoderBinaryPair::Decode(
          static_cast<uint32_t>(start_row_input), static_cast<uint32_t>(num_rows),
          row_metadata_.column_offsets[i], rows, &batch_all_cols_[i],
          &batch_all_cols_[i + 1], &ctx, &temp_buffer_A, &temp_buffer_B);
      i += 2;
    }
  }

  // Process nulls
  EncoderNulls::Decode(static_cast<uint32_t>(start_row_input),
                       static_cast<uint32_t>(num_rows), rows, &batch_all_cols_);
}

void RowEncoder::DecodeVaryingLengthBuffers(int64_t start_row_input,
                                            int64_t start_row_output, int64_t num_rows,
                                            const RowTable& rows,
                                            std::vector<KeyColumnArray>* cols,
                                            int64_t hardware_flags,
                                            util::TempVectorStack* temp_stack) {
  // Prepare column array vectors
  PrepareKeyColumnArrays(start_row_output, num_rows, *cols);

  RowEncoderContext ctx;
  ctx.hardware_flags = hardware_flags;
  ctx.stack = temp_stack;

  bool is_row_fixed_length = row_metadata_.is_fixed_length;
  if (!is_row_fixed_length) {
    for (size_t i = 0; i < batch_varbinary_cols_.size(); ++i) {
      // Memcpy varbinary fields into precomputed in the previous step
      // positions in the output row buffer.
      EncoderVarBinary::Decode(static_cast<uint32_t>(start_row_input),
                               static_cast<uint32_t>(num_rows), static_cast<uint32_t>(i),
                               rows, &batch_varbinary_cols_[i], &ctx);
    }
  }
}

template <class COPY_FN, class SET_NULL_FN>
void RowEncoder::EncoderBinary::EncodeSelectedImp(
    uint32_t offset_within_row, RowTable* rows, const KeyColumnArray& col,
    uint32_t num_selected, const uint16_t* selection, COPY_FN copy_fn,
    SET_NULL_FN set_null_fn) {
  bool is_fixed_length = rows->metadata().is_fixed_length;
  if (is_fixed_length) {
    uint32_t row_width = rows->metadata().fixed_length;
    const uint8_t* src_base = col.data(1);
    uint8_t* dst = rows->mutable_data(1) + offset_within_row;
    for (uint32_t i = 0; i < num_selected; ++i) {
      copy_fn(dst, src_base, selection[i]);
      dst += row_width;
    }
    if (col.data(0)) {
      const uint8_t* non_null_bits = col.data(0);
      uint8_t* dst = rows->mutable_data(1) + offset_within_row;
      for (uint32_t i = 0; i < num_selected; ++i) {
        bool is_null = !bit_util::GetBit(non_null_bits, selection[i] + col.bit_offset(0));
        if (is_null) {
          set_null_fn(dst);
        }
        dst += row_width;
      }
    }
  } else {
    const uint8_t* src_base = col.data(1);
    uint8_t* dst = rows->mutable_data(2) + offset_within_row;
    const uint32_t* offsets = rows->offsets();
    for (uint32_t i = 0; i < num_selected; ++i) {
      copy_fn(dst + offsets[i], src_base, selection[i]);
    }
    if (col.data(0)) {
      const uint8_t* non_null_bits = col.data(0);
      uint8_t* dst = rows->mutable_data(2) + offset_within_row;
      const uint32_t* offsets = rows->offsets();
      for (uint32_t i = 0; i < num_selected; ++i) {
        bool is_null = !bit_util::GetBit(non_null_bits, selection[i] + col.bit_offset(0));
        if (is_null) {
          set_null_fn(dst + offsets[i]);
        }
      }
    }
  }
}

void RowEncoder::EncoderBinary::EncodeSelected(uint32_t offset_within_row, RowTable* rows,
                                               const KeyColumnArray& col,
                                               uint32_t num_selected,
                                               const uint16_t* selection) {
  if (col.metadata().is_null_type) {
    return;
  }
  uint32_t col_width = col.metadata().fixed_length;
  if (col_width == 0) {
    int bit_offset = col.bit_offset(1);
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [bit_offset](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *dst = bit_util::GetBit(src_base, irow + bit_offset) ? 0xff : 0x00;
        },
        [](uint8_t* dst) { *dst = 0xae; });
  } else if (col_width == 1) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *dst = src_base[irow];
        },
        [](uint8_t* dst) { *dst = 0xae; });
  } else if (col_width == 2) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *reinterpret_cast<uint16_t*>(dst) =
              reinterpret_cast<const uint16_t*>(src_base)[irow];
        },
        [](uint8_t* dst) { *reinterpret_cast<uint16_t*>(dst) = 0xaeae; });
  } else if (col_width == 4) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *reinterpret_cast<uint32_t*>(dst) =
              reinterpret_cast<const uint32_t*>(src_base)[irow];
        },
        [](uint8_t* dst) {
          *reinterpret_cast<uint32_t*>(dst) = static_cast<uint32_t>(0xaeaeaeae);
        });
  } else if (col_width == 8) {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          *reinterpret_cast<uint64_t*>(dst) =
              reinterpret_cast<const uint64_t*>(src_base)[irow];
        },
        [](uint8_t* dst) { *reinterpret_cast<uint64_t*>(dst) = 0xaeaeaeaeaeaeaeaeULL; });
  } else {
    EncodeSelectedImp(
        offset_within_row, rows, col, num_selected, selection,
        [col_width](uint8_t* dst, const uint8_t* src_base, uint16_t irow) {
          memcpy(dst, src_base + col_width * irow, col_width);
        },
        [col_width](uint8_t* dst) { memset(dst, 0xae, col_width); });
  }
}

void RowEncoder::EncoderOffsets::GetRowOffsetsSelected(
    RowTable* rows, const std::vector<KeyColumnArray>& cols, uint32_t num_selected,
    const uint16_t* selection) {
  if (rows->metadata().is_fixed_length) {
    return;
  }

  uint32_t* row_offsets = rows->mutable_offsets();
  for (uint32_t i = 0; i < num_selected; ++i) {
    row_offsets[i] = rows->metadata().fixed_length;
  }

  for (size_t icol = 0; icol < cols.size(); ++icol) {
    bool is_fixed_length = (cols[icol].metadata().is_fixed_length);
    if (!is_fixed_length) {
      const uint32_t* col_offsets = cols[icol].offsets();
      for (uint32_t i = 0; i < num_selected; ++i) {
        uint32_t irow = selection[i];
        uint32_t length = col_offsets[irow + 1] - col_offsets[irow];
        row_offsets[i] += RowTableMetadata::padding_for_alignment(
            row_offsets[i], rows->metadata().string_alignment);
        row_offsets[i] += length;
      }
      const uint8_t* non_null_bits = cols[icol].data(0);
      if (non_null_bits) {
        const uint32_t* col_offsets = cols[icol].offsets();
        for (uint32_t i = 0; i < num_selected; ++i) {
          uint32_t irow = selection[i];
          bool is_null =
              !bit_util::GetBit(non_null_bits, irow + cols[icol].bit_offset(0));
          if (is_null) {
            uint32_t length = col_offsets[irow + 1] - col_offsets[irow];
            row_offsets[i] -= length;
          }
        }
      }
    }
  }

  uint32_t sum = 0;
  int row_alignment = rows->metadata().row_alignment;
  for (uint32_t i = 0; i < num_selected; ++i) {
    uint32_t length = row_offsets[i];
    length += RowTableMetadata::padding_for_alignment(length, row_alignment);
    row_offsets[i] = sum;
    sum += length;
  }
  row_offsets[num_selected] = sum;
}

template <bool has_nulls, bool is_first_varbinary>
void RowEncoder::EncoderOffsets::EncodeSelectedImp(
    uint32_t ivarbinary, RowTable* rows, const std::vector<KeyColumnArray>& cols,
    uint32_t num_selected, const uint16_t* selection) {
  const uint32_t* row_offsets = rows->offsets();
  uint8_t* row_base = rows->mutable_data(2) +
                      rows->metadata().varbinary_end_array_offset +
                      ivarbinary * sizeof(uint32_t);
  const uint32_t* col_offsets = cols[ivarbinary].offsets();
  const uint8_t* col_non_null_bits = cols[ivarbinary].data(0);

  for (uint32_t i = 0; i < num_selected; ++i) {
    uint32_t irow = selection[i];
    uint32_t length = col_offsets[irow + 1] - col_offsets[irow];
    if (has_nulls) {
      uint32_t null_multiplier =
          bit_util::GetBit(col_non_null_bits, irow + cols[ivarbinary].bit_offset(0)) ? 1
                                                                                     : 0;
      length *= null_multiplier;
    }
    uint32_t* row = reinterpret_cast<uint32_t*>(row_base + row_offsets[i]);
    if (is_first_varbinary) {
      row[0] = rows->metadata().fixed_length + length;
    } else {
      row[0] = row[-1] +
               RowTableMetadata::padding_for_alignment(
                   row[-1], rows->metadata().string_alignment) +
               length;
    }
  }
}

void RowEncoder::EncoderOffsets::EncodeSelected(RowTable* rows,
                                                const std::vector<KeyColumnArray>& cols,
                                                uint32_t num_selected,
                                                const uint16_t* selection) {
  if (rows->metadata().is_fixed_length) {
    return;
  }
  uint32_t ivarbinary = 0;
  for (size_t icol = 0; icol < cols.size(); ++icol) {
    if (!cols[icol].metadata().is_fixed_length) {
      const uint8_t* non_null_bits = cols[icol].data(0);
      if (non_null_bits && ivarbinary == 0) {
        EncodeSelectedImp<true, true>(ivarbinary, rows, cols, num_selected, selection);
      } else if (non_null_bits && ivarbinary > 0) {
        EncodeSelectedImp<true, false>(ivarbinary, rows, cols, num_selected, selection);
      } else if (!non_null_bits && ivarbinary == 0) {
        EncodeSelectedImp<false, true>(ivarbinary, rows, cols, num_selected, selection);
      } else {
        EncodeSelectedImp<false, false>(ivarbinary, rows, cols, num_selected, selection);
      }
      ivarbinary++;
    }
  }
}

void RowEncoder::EncoderVarBinary::EncodeSelected(uint32_t ivarbinary, RowTable* rows,
                                                  const KeyColumnArray& cols,
                                                  uint32_t num_selected,
                                                  const uint16_t* selection) {
  const uint32_t* row_offsets = rows->offsets();
  uint8_t* row_base = rows->mutable_data(2);
  const uint32_t* col_offsets = cols.offsets();
  const uint8_t* col_base = cols.data(2);

  if (ivarbinary == 0) {
    for (uint32_t i = 0; i < num_selected; ++i) {
      uint8_t* row = row_base + row_offsets[i];
      uint32_t row_offset;
      uint32_t length;
      rows->metadata().first_varbinary_offset_and_length(row, &row_offset, &length);
      uint32_t irow = selection[i];
      memcpy(row + row_offset, col_base + col_offsets[irow], length);
    }
  } else {
    for (uint32_t i = 0; i < num_selected; ++i) {
      uint8_t* row = row_base + row_offsets[i];
      uint32_t row_offset;
      uint32_t length;
      rows->metadata().nth_varbinary_offset_and_length(row, ivarbinary, &row_offset,
                                                       &length);
      uint32_t irow = selection[i];
      memcpy(row + row_offset, col_base + col_offsets[irow], length);
    }
  }
}

void RowEncoder::EncoderNulls::EncodeSelected(RowTable* rows,
                                              const std::vector<KeyColumnArray>& cols,
                                              uint32_t num_selected,
                                              const uint16_t* selection) {
  uint8_t* null_masks = rows->null_masks();
  uint32_t null_mask_num_bytes = rows->metadata().null_masks_bytes_per_row;
  memset(null_masks, 0, null_mask_num_bytes * num_selected);
  for (size_t icol = 0; icol < cols.size(); ++icol) {
    const uint8_t* non_null_bits = cols[icol].data(0);
    if (non_null_bits) {
      for (uint32_t i = 0; i < num_selected; ++i) {
        uint32_t irow = selection[i];
        bool is_null = !bit_util::GetBit(non_null_bits, irow + cols[icol].bit_offset(0));
        if (is_null) {
          bit_util::SetBit(null_masks, i * null_mask_num_bytes * 8 + icol);
        }
      }
    }
  }
}

void RowEncoder::PrepareEncodeSelected(int64_t start_row, int64_t num_rows,
                                       const std::vector<KeyColumnArray>& cols) {
  // Prepare column array vectors
  PrepareKeyColumnArrays(start_row, num_rows, cols);
}

Status RowEncoder::EncodeSelected(RowTable* rows, uint32_t num_selected,
                                  const uint16_t* selection) {
  rows->Clean();
  RETURN_NOT_OK(
      rows->AppendEmpty(static_cast<uint32_t>(num_selected), static_cast<uint32_t>(0)));

  EncoderOffsets::GetRowOffsetsSelected(rows, batch_varbinary_cols_, num_selected,
                                        selection);

  RETURN_NOT_OK(rows->AppendEmpty(static_cast<uint32_t>(0),
                                  static_cast<uint32_t>(rows->offsets()[num_selected])));

  for (size_t icol = 0; icol < batch_all_cols_.size(); ++icol) {
    if (batch_all_cols_[icol].metadata().is_fixed_length) {
      uint32_t offset_within_row = rows->metadata().column_offsets[icol];
      EncoderBinary::EncodeSelected(offset_within_row, rows, batch_all_cols_[icol],
                                    num_selected, selection);
    }
  }

  EncoderOffsets::EncodeSelected(rows, batch_varbinary_cols_, num_selected, selection);

  for (size_t icol = 0; icol < batch_varbinary_cols_.size(); ++icol) {
    EncoderVarBinary::EncodeSelected(static_cast<uint32_t>(icol), rows,
                                     batch_varbinary_cols_[icol], num_selected,
                                     selection);
  }

  EncoderNulls::EncodeSelected(rows, batch_all_cols_, num_selected, selection);

  return Status::OK();
}

Status RowArray::InitIfNeeded(MemoryPool* pool,
                              const RowEncoder::RowTableMetadata& row_metadata) {
  if (is_initialized_) {
    return Status::OK();
  }
  encoder_.Init(row_metadata.column_metadatas, sizeof(uint64_t), sizeof(uint64_t));
  RETURN_NOT_OK(rows_temp_.Init(pool, row_metadata));
  RETURN_NOT_OK(rows_.Init(pool, row_metadata));
  is_initialized_ = true;
  return Status::OK();
}

Status RowArray::InitIfNeeded(MemoryPool* pool, const ExecBatch& batch) {
  if (is_initialized_) {
    return Status::OK();
  }
  std::vector<KeyColumnMetadata> column_metadatas;
  RETURN_NOT_OK(ColumnMetadatasFromExecBatch(batch, &column_metadatas));
  RowEncoder::RowTableMetadata row_metadata;
  row_metadata.FromColumnMetadataVector(column_metadatas, sizeof(uint64_t),
                                        sizeof(uint64_t));

  return InitIfNeeded(pool, row_metadata);
}

Status RowArray::AppendBatchSelection(MemoryPool* pool, const ExecBatch& batch,
                                      int begin_row_id, int end_row_id, int num_row_ids,
                                      const uint16_t* row_ids,
                                      std::vector<KeyColumnArray>& temp_column_arrays) {
  RETURN_NOT_OK(InitIfNeeded(pool, batch));
  RETURN_NOT_OK(ColumnArraysFromExecBatch(batch, begin_row_id, end_row_id - begin_row_id,
                                          &temp_column_arrays));
  encoder_.PrepareEncodeSelected(
      /*start_row=*/0, end_row_id - begin_row_id, temp_column_arrays);
  RETURN_NOT_OK(encoder_.EncodeSelected(&rows_temp_, num_row_ids, row_ids));
  RETURN_NOT_OK(rows_.AppendSelectionFrom(rows_temp_, num_row_ids, nullptr));
  return Status::OK();
}

void RowArray::Compare(const ExecBatch& batch, int begin_row_id, int end_row_id,
                       int num_selected, const uint16_t* batch_selection_maybe_null,
                       const uint32_t* array_row_ids, uint32_t* out_num_not_equal,
                       uint16_t* out_not_equal_selection, int64_t hardware_flags,
                       util::TempVectorStack* temp_stack,
                       std::vector<KeyColumnArray>& temp_column_arrays,
                       uint8_t* out_match_bitvector_maybe_null) {
  Status status = ColumnArraysFromExecBatch(
      batch, begin_row_id, end_row_id - begin_row_id, &temp_column_arrays);
  ARROW_DCHECK(status.ok());

  LightContext ctx;
  ctx.hardware_flags = hardware_flags;
  ctx.stack = temp_stack;
  KeyCompare::CompareColumnsToRows(
      num_selected, batch_selection_maybe_null, array_row_ids, &ctx, out_num_not_equal,
      out_not_equal_selection, temp_column_arrays, rows_,
      /*are_cols_in_encoding_order=*/false, out_match_bitvector_maybe_null);
}

Status RowArray::DecodeSelected(ResizableArrayData* output, int column_id,
                                int num_rows_to_append, const uint32_t* row_ids,
                                MemoryPool* pool) const {
  int num_rows_before = output->num_rows();
  RETURN_NOT_OK(output->ResizeFixedLengthBuffers(num_rows_before + num_rows_to_append));

  // Both input (RowTable) and output (ResizableArrayData) have buffers with
  // extra bytes added at the end to avoid buffer overruns when using wide load
  // instructions.
  //

  ARROW_ASSIGN_OR_RAISE(KeyColumnMetadata column_metadata, output->column_metadata());

  if (column_metadata.is_fixed_length) {
    uint32_t fixed_length = column_metadata.fixed_length;
    switch (fixed_length) {
      case 0:
        RowArrayAccessor::Visit(rows_, column_id, num_rows_to_append, row_ids,
                                [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
                                  bit_util::SetBitTo(output->mutable_data(1),
                                                     num_rows_before + i, *ptr != 0);
                                });
        break;
      case 1:
        RowArrayAccessor::Visit(rows_, column_id, num_rows_to_append, row_ids,
                                [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
                                  output->mutable_data(1)[num_rows_before + i] = *ptr;
                                });
        break;
      case 2:
        RowArrayAccessor::Visit(
            rows_, column_id, num_rows_to_append, row_ids,
            [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
              reinterpret_cast<uint16_t*>(output->mutable_data(1))[num_rows_before + i] =
                  *reinterpret_cast<const uint16_t*>(ptr);
            });
        break;
      case 4:
        RowArrayAccessor::Visit(
            rows_, column_id, num_rows_to_append, row_ids,
            [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
              reinterpret_cast<uint32_t*>(output->mutable_data(1))[num_rows_before + i] =
                  *reinterpret_cast<const uint32_t*>(ptr);
            });
        break;
      case 8:
        RowArrayAccessor::Visit(
            rows_, column_id, num_rows_to_append, row_ids,
            [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
              reinterpret_cast<uint64_t*>(output->mutable_data(1))[num_rows_before + i] =
                  *reinterpret_cast<const uint64_t*>(ptr);
            });
        break;
      default:
        RowArrayAccessor::Visit(
            rows_, column_id, num_rows_to_append, row_ids,
            [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
              uint64_t* dst = reinterpret_cast<uint64_t*>(
                  output->mutable_data(1) + num_bytes * (num_rows_before + i));
              const uint64_t* src = reinterpret_cast<const uint64_t*>(ptr);
              for (uint32_t word_id = 0;
                   word_id < bit_util::CeilDiv(num_bytes, sizeof(uint64_t)); ++word_id) {
                util::SafeStore<uint64_t>(dst + word_id, util::SafeLoad(src + word_id));
              }
            });
        break;
    }
  } else {
    uint32_t* offsets =
        reinterpret_cast<uint32_t*>(output->mutable_data(1)) + num_rows_before;
    uint32_t sum = num_rows_before == 0 ? 0 : offsets[0];
    RowArrayAccessor::Visit(
        rows_, column_id, num_rows_to_append, row_ids,
        [&](int i, const uint8_t* ptr, uint32_t num_bytes) { offsets[i] = num_bytes; });
    for (int i = 0; i < num_rows_to_append; ++i) {
      uint32_t length = offsets[i];
      offsets[i] = sum;
      sum += length;
    }
    offsets[num_rows_to_append] = sum;
    RETURN_NOT_OK(output->ResizeVaryingLengthBuffer());
    RowArrayAccessor::Visit(
        rows_, column_id, num_rows_to_append, row_ids,
        [&](int i, const uint8_t* ptr, uint32_t num_bytes) {
          uint64_t* dst = reinterpret_cast<uint64_t*>(
              output->mutable_data(2) +
              reinterpret_cast<const uint32_t*>(
                  output->mutable_data(1))[num_rows_before + i]);
          const uint64_t* src = reinterpret_cast<const uint64_t*>(ptr);
          for (uint32_t word_id = 0;
               word_id < bit_util::CeilDiv(num_bytes, sizeof(uint64_t)); ++word_id) {
            util::SafeStore<uint64_t>(dst + word_id, util::SafeLoad(src + word_id));
          }
        });
  }

  // Process nulls
  //
  RowArrayAccessor::VisitNulls(
      rows_, column_id, num_rows_to_append, row_ids, [&](int i, uint8_t value) {
        bit_util::SetBitTo(output->mutable_data(0), num_rows_before + i, value == 0);
      });

  return Status::OK();
}

Status RowArrayMerge::PrepareForMerge(RowArray* target,
                                      const std::vector<RowArray*>& sources,
                                      std::vector<int64_t>* first_target_row_id,
                                      MemoryPool* pool) {
  ARROW_DCHECK(!sources.empty());

  ARROW_DCHECK(sources[0]->is_initialized_);
  const RowEncoder::RowTableMetadata& metadata = sources[0]->rows_.metadata();
  ARROW_DCHECK(!target->is_initialized_);
  RETURN_NOT_OK(target->InitIfNeeded(pool, metadata));

  // Sum the number of rows from all input sources and calculate their total
  // size.
  //
  int64_t num_rows = 0;
  int64_t num_bytes = 0;
  first_target_row_id->resize(sources.size() + 1);
  for (size_t i = 0; i < sources.size(); ++i) {
    // All input sources must be initialized and have the same row format.
    //
    ARROW_DCHECK(sources[i]->is_initialized_);
    ARROW_DCHECK(metadata.is_compatible(sources[i]->rows_.metadata()));
    (*first_target_row_id)[i] = num_rows;
    num_rows += sources[i]->rows_.length();
    if (!metadata.is_fixed_length) {
      num_bytes += sources[i]->rows_.offsets()[sources[i]->rows_.length()];
    }
  }
  (*first_target_row_id)[sources.size()] = num_rows;

  // Allocate target memory
  //
  target->rows_.Clean();
  RETURN_NOT_OK(target->rows_.AppendEmpty(static_cast<uint32_t>(num_rows),
                                          static_cast<uint32_t>(num_bytes)));

  // In case of varying length rows,
  // initialize the first row offset for each range of rows corresponding to a
  // single source.
  //
  if (!metadata.is_fixed_length) {
    num_rows = 0;
    num_bytes = 0;
    for (size_t i = 0; i < sources.size(); ++i) {
      target->rows_.mutable_offsets()[num_rows] = static_cast<uint32_t>(num_bytes);
      num_rows += sources[i]->rows_.length();
      num_bytes += sources[i]->rows_.offsets()[sources[i]->rows_.length()];
    }
    target->rows_.mutable_offsets()[num_rows] = static_cast<uint32_t>(num_bytes);
  }

  return Status::OK();
}

void RowArrayMerge::MergeSingle(RowArray* target, const RowArray& source,
                                int64_t first_target_row_id,
                                const int64_t* source_rows_permutation) {
  // Source and target must:
  // - be initialized
  // - use the same row format
  // - use 64-bit alignment
  //
  ARROW_DCHECK(source.is_initialized_ && target->is_initialized_);
  ARROW_DCHECK(target->rows_.metadata().is_compatible(source.rows_.metadata()));
  ARROW_DCHECK(target->rows_.metadata().row_alignment == sizeof(uint64_t));

  if (target->rows_.metadata().is_fixed_length) {
    CopyFixedLength(&target->rows_, source.rows_, first_target_row_id,
                    source_rows_permutation);
  } else {
    CopyVaryingLength(&target->rows_, source.rows_, first_target_row_id,
                      target->rows_.offsets()[first_target_row_id],
                      source_rows_permutation);
  }
  CopyNulls(&target->rows_, source.rows_, first_target_row_id, source_rows_permutation);
}

void RowArrayMerge::CopyFixedLength(RowEncoder::RowTable* target,
                                    const RowEncoder::RowTable& source,
                                    int64_t first_target_row_id,
                                    const int64_t* source_rows_permutation) {
  int64_t num_source_rows = source.length();

  int64_t fixed_length = target->metadata().fixed_length;

  // Permutation of source rows is optional. Without permutation all that is
  // needed is memcpy.
  //
  if (!source_rows_permutation) {
    memcpy(target->mutable_data(1) + fixed_length * first_target_row_id, source.data(1),
           fixed_length * num_source_rows);
  } else {
    // Row length must be a multiple of 64-bits due to enforced alignment.
    // Loop for each output row copying a fixed number of 64-bit words.
    //
    ARROW_DCHECK(fixed_length % sizeof(uint64_t) == 0);

    int64_t num_words_per_row = fixed_length / sizeof(uint64_t);
    for (int64_t i = 0; i < num_source_rows; ++i) {
      int64_t source_row_id = source_rows_permutation[i];
      const uint64_t* source_row_ptr = reinterpret_cast<const uint64_t*>(
          source.data(1) + fixed_length * source_row_id);
      uint64_t* target_row_ptr = reinterpret_cast<uint64_t*>(
          target->mutable_data(1) + fixed_length * (first_target_row_id + i));

      for (int64_t word = 0; word < num_words_per_row; ++word) {
        target_row_ptr[word] = source_row_ptr[word];
      }
    }
  }
}

void RowArrayMerge::CopyVaryingLength(RowEncoder::RowTable* target,
                                      const RowEncoder::RowTable& source,
                                      int64_t first_target_row_id,
                                      int64_t first_target_row_offset,
                                      const int64_t* source_rows_permutation) {
  int64_t num_source_rows = source.length();
  uint32_t* target_offsets = target->mutable_offsets();
  const uint32_t* source_offsets = source.offsets();

  // Permutation of source rows is optional.
  //
  if (!source_rows_permutation) {
    int64_t target_row_offset = first_target_row_offset;
    for (int64_t i = 0; i < num_source_rows; ++i) {
      target_offsets[first_target_row_id + i] = static_cast<uint32_t>(target_row_offset);
      target_row_offset += source_offsets[i + 1] - source_offsets[i];
    }
    // We purposefully skip outputting of N+1 offset, to allow concurrent
    // copies of rows done to adjacent ranges in target array.
    // It should have already been initialized during preparation for merge.
    //

    // We can simply memcpy bytes of rows if their order has not changed.
    //
    memcpy(target->mutable_data(2) + target_offsets[first_target_row_id], source.data(2),
           source_offsets[num_source_rows] - source_offsets[0]);
  } else {
    int64_t target_row_offset = first_target_row_offset;
    uint64_t* target_row_ptr =
        reinterpret_cast<uint64_t*>(target->mutable_data(2) + target_row_offset);
    for (int64_t i = 0; i < num_source_rows; ++i) {
      int64_t source_row_id = source_rows_permutation[i];
      const uint64_t* source_row_ptr = reinterpret_cast<const uint64_t*>(
          source.data(2) + source_offsets[source_row_id]);
      uint32_t length = source_offsets[source_row_id + 1] - source_offsets[source_row_id];

      // Rows should be 64-bit aligned.
      // In that case we can copy them using a sequence of 64-bit read/writes.
      //
      ARROW_DCHECK(length % sizeof(uint64_t) == 0);

      for (uint32_t word = 0; word < length / sizeof(uint64_t); ++word) {
        *target_row_ptr++ = *source_row_ptr++;
      }

      target_offsets[first_target_row_id + i] = static_cast<uint32_t>(target_row_offset);
      target_row_offset += length;
    }
  }
}

void RowArrayMerge::CopyNulls(RowEncoder::RowTable* target,
                              const RowEncoder::RowTable& source,
                              int64_t first_target_row_id,
                              const int64_t* source_rows_permutation) {
  int64_t num_source_rows = source.length();
  int num_bytes_per_row = target->metadata().null_masks_bytes_per_row;
  uint8_t* target_nulls = target->null_masks() + num_bytes_per_row * first_target_row_id;
  if (!source_rows_permutation) {
    memcpy(target_nulls, source.null_masks(), num_bytes_per_row * num_source_rows);
  } else {
    for (int64_t i = 0; i < num_source_rows; ++i) {
      int64_t source_row_id = source_rows_permutation[i];
      const uint8_t* source_nulls =
          source.null_masks() + num_bytes_per_row * source_row_id;
      for (int64_t byte = 0; byte < num_bytes_per_row; ++byte) {
        *target_nulls++ = *source_nulls++;
      }
    }
  }
}

int RowArrayAccessor::VarbinaryColumnId(const RowEncoder::RowTableMetadata& row_metadata,
                                        int column_id) {
  ARROW_DCHECK(row_metadata.num_cols() > static_cast<uint32_t>(column_id));
  ARROW_DCHECK(!row_metadata.is_fixed_length);
  ARROW_DCHECK(!row_metadata.column_metadatas[column_id].is_fixed_length);

  int varbinary_column_id = 0;
  for (int i = 0; i < column_id; ++i) {
    if (!row_metadata.column_metadatas[i].is_fixed_length) {
      ++varbinary_column_id;
    }
  }
  return varbinary_column_id;
}

int RowArrayAccessor::NumRowsToSkip(const RowEncoder::RowTable& rows, int column_id,
                                    int num_rows, const uint32_t* row_ids,
                                    int num_tail_bytes_to_skip) {
  uint32_t num_bytes_skipped = 0;
  int num_rows_left = num_rows;

  bool is_fixed_length_column =
      rows.metadata().column_metadatas[column_id].is_fixed_length;

  if (!is_fixed_length_column) {
    // Varying length column
    //
    int varbinary_column_id = VarbinaryColumnId(rows.metadata(), column_id);

    while (num_rows_left > 0 &&
           num_bytes_skipped < static_cast<uint32_t>(num_tail_bytes_to_skip)) {
      // Find the pointer to the last requested row
      //
      uint32_t last_row_id = row_ids[num_rows_left - 1];
      const uint8_t* row_ptr = rows.data(2) + rows.offsets()[last_row_id];

      // Find the length of the requested varying length field in that row
      //
      uint32_t field_offset_within_row, field_length;
      if (varbinary_column_id == 0) {
        rows.metadata().first_varbinary_offset_and_length(
            row_ptr, &field_offset_within_row, &field_length);
      } else {
        rows.metadata().nth_varbinary_offset_and_length(
            row_ptr, varbinary_column_id, &field_offset_within_row, &field_length);
      }

      num_bytes_skipped += field_length;
      --num_rows_left;
    }
  } else {
    // Fixed length column
    //
    uint32_t field_length = rows.metadata().column_metadatas[column_id].fixed_length;
    uint32_t num_bytes_skipped = 0;
    while (num_rows_left > 0 &&
           num_bytes_skipped < static_cast<uint32_t>(num_tail_bytes_to_skip)) {
      num_bytes_skipped += field_length;
      --num_rows_left;
    }
  }

  return num_rows - num_rows_left;
}

template <class PROCESS_VALUE_FN>
void RowArrayAccessor::Visit(const RowEncoder::RowTable& rows, int column_id,
                             int num_rows, const uint32_t* row_ids,
                             PROCESS_VALUE_FN process_value_fn) {
  bool is_fixed_length_column =
      rows.metadata().column_metadatas[column_id].is_fixed_length;

  // There are 4 cases, each requiring different steps:
  // 1. Varying length column that is the first varying length column in a row
  // 2. Varying length column that is not the first varying length column in a
  // row
  // 3. Fixed length column in a fixed length row
  // 4. Fixed length column in a varying length row

  if (!is_fixed_length_column) {
    int varbinary_column_id = VarbinaryColumnId(rows.metadata(), column_id);
    const uint8_t* row_ptr_base = rows.data(2);
    const uint32_t* row_offsets = rows.offsets();
    uint32_t field_offset_within_row, field_length;

    if (varbinary_column_id == 0) {
      // Case 1: This is the first varbinary column
      //
      for (int i = 0; i < num_rows; ++i) {
        uint32_t row_id = row_ids[i];
        const uint8_t* row_ptr = row_ptr_base + row_offsets[row_id];
        rows.metadata().first_varbinary_offset_and_length(
            row_ptr, &field_offset_within_row, &field_length);
        process_value_fn(i, row_ptr + field_offset_within_row, field_length);
      }
    } else {
      // Case 2: This is second or later varbinary column
      //
      for (int i = 0; i < num_rows; ++i) {
        uint32_t row_id = row_ids[i];
        const uint8_t* row_ptr = row_ptr_base + row_offsets[row_id];
        rows.metadata().nth_varbinary_offset_and_length(
            row_ptr, varbinary_column_id, &field_offset_within_row, &field_length);
        process_value_fn(i, row_ptr + field_offset_within_row, field_length);
      }
    }
  }

  if (is_fixed_length_column) {
    uint32_t field_offset_within_row = rows.metadata().encoded_field_offset(
        rows.metadata().pos_after_encoding(column_id));
    uint32_t field_length = rows.metadata().column_metadatas[column_id].fixed_length;
    // Bit column is encoded as a single byte
    //
    if (field_length == 0) {
      field_length = 1;
    }
    uint32_t row_length = rows.metadata().fixed_length;

    bool is_fixed_length_row = rows.metadata().is_fixed_length;
    if (is_fixed_length_row) {
      // Case 3: This is a fixed length column in a fixed length row
      //
      const uint8_t* row_ptr_base = rows.data(1) + field_offset_within_row;
      for (int i = 0; i < num_rows; ++i) {
        uint32_t row_id = row_ids[i];
        const uint8_t* row_ptr = row_ptr_base + row_length * row_id;
        process_value_fn(i, row_ptr, field_length);
      }
    } else {
      // Case 4: This is a fixed length column in a varying length row
      //
      const uint8_t* row_ptr_base = rows.data(2) + field_offset_within_row;
      const uint32_t* row_offsets = rows.offsets();
      for (int i = 0; i < num_rows; ++i) {
        uint32_t row_id = row_ids[i];
        const uint8_t* row_ptr = row_ptr_base + row_offsets[row_id];
        process_value_fn(i, row_ptr, field_length);
      }
    }
  }
}

template <class PROCESS_VALUE_FN>
void RowArrayAccessor::VisitNulls(const RowEncoder::RowTable& rows, int column_id,
                                  int num_rows, const uint32_t* row_ids,
                                  PROCESS_VALUE_FN process_value_fn) {
  const uint8_t* null_masks = rows.null_masks();
  uint32_t null_mask_num_bytes = rows.metadata().null_masks_bytes_per_row;
  uint32_t pos_after_encoding = rows.metadata().pos_after_encoding(column_id);
  for (int i = 0; i < num_rows; ++i) {
    uint32_t row_id = row_ids[i];
    int64_t bit_id = row_id * null_mask_num_bytes * 8 + pos_after_encoding;
    process_value_fn(i, bit_util::GetBit(null_masks, bit_id) ? 0xff : 0);
  }
}

}  // namespace
}  // namespace compute
