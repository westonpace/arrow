#include "arrow/compute/row/encode_internal.h"

namespace arrow {
namespace compute {

namespace {
struct TransformBoolean {
  static KeyColumnArray ArrayReplace(const KeyColumnArray& column,
                                     const KeyColumnArray& temp) {
    // Make sure that the temp buffer is large enough
    DCHECK(temp.length() >= column.length() && temp.metadata().is_fixed_length &&
           temp.metadata().fixed_length >= sizeof(uint8_t));
    KeyColumnMetadata metadata;
    metadata.is_fixed_length = true;
    metadata.fixed_length = sizeof(uint8_t);
    constexpr int buffer_index = 1;
    return column.WithBufferFrom(temp, buffer_index).WithMetadata(metadata);
  }

  static void PostDecode(const KeyColumnArray& input, KeyColumnArray* output,
                         LightContext* ctx) {
    // Make sure that metadata and lengths are compatible.
    DCHECK(output->metadata().is_fixed_length == input.metadata().is_fixed_length);
    DCHECK(output->metadata().fixed_length == 0 && input.metadata().fixed_length == 1);
    DCHECK(output->length() == input.length());
    constexpr int buffer_index = 1;
    DCHECK(input.data(buffer_index) != nullptr);
    DCHECK(output->mutable_data(buffer_index) != nullptr);

    util::bit_util::bytes_to_bits(
        ctx->hardware_flags, static_cast<int>(input.length()), input.data(buffer_index),
        output->mutable_data(buffer_index), output->bit_offset(buffer_index));
  }
};

}  // namespace

bool EncoderInteger::IsBoolean(const KeyColumnMetadata& metadata) {
  return metadata.is_fixed_length && metadata.fixed_length == 0 && !metadata.is_null_type;
}

bool EncoderInteger::UsesTransform(const KeyColumnArray& column) {
  return IsBoolean(column.metadata());
}

KeyColumnArray EncoderInteger::ArrayReplace(const KeyColumnArray& column,
                                                        const KeyColumnArray& temp) {
  if (IsBoolean(column.metadata())) {
    return TransformBoolean::ArrayReplace(column, temp);
  }
  return column;
}

void EncoderInteger::PostDecode(const KeyColumnArray& input,
                                            KeyColumnArray* output,
                                            LightContext* ctx) {
  if (IsBoolean(output->metadata())) {
    TransformBoolean::PostDecode(input, output, ctx);
  }
}

void EncoderInteger::Decode(uint32_t start_row, uint32_t num_rows,
                                        uint32_t offset_within_row, const RowTable& rows,
                                        KeyColumnArray* col, LightContext* ctx,
                                        KeyColumnArray* temp) {
  KeyColumnArray col_prep;
  if (UsesTransform(*col)) {
    col_prep = ArrayReplace(*col, *temp);
  } else {
    col_prep = *col;
  }

  // When we have a single fixed length column we can just do memcpy
  if (rows.metadata().is_fixed_length &&
      col_prep.metadata().fixed_length == rows.metadata().fixed_length) {
    DCHECK_EQ(offset_within_row, 0);
    uint32_t row_size = rows.metadata().fixed_length;
    memcpy(col_prep.mutable_data(1), rows.data(1) + start_row * row_size,
           num_rows * row_size);
  } else if (rows.metadata().is_fixed_length) {
    uint32_t row_size = rows.metadata().fixed_length;
    const uint8_t* row_base = rows.data(1) + start_row * row_size;
    row_base += offset_within_row;
    uint8_t* col_base = col_prep.mutable_data(1);
    switch (col_prep.metadata().fixed_length) {
      case 1:
        for (uint32_t i = 0; i < num_rows; ++i) {
          col_base[i] = row_base[i * row_size];
        }
        break;
      case 2:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint16_t*>(col_base)[i] =
              *reinterpret_cast<const uint16_t*>(row_base + i * row_size);
        }
        break;
      case 4:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint32_t*>(col_base)[i] =
              *reinterpret_cast<const uint32_t*>(row_base + i * row_size);
        }
        break;
      case 8:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint64_t*>(col_base)[i] =
              *reinterpret_cast<const uint64_t*>(row_base + i * row_size);
        }
        break;
      default:
        DCHECK(false);
    }
  } else {
    const uint32_t* row_offsets = rows.offsets() + start_row;
    const uint8_t* row_base = rows.data(2);
    row_base += offset_within_row;
    uint8_t* col_base = col_prep.mutable_data(1);
    switch (col_prep.metadata().fixed_length) {
      case 1:
        for (uint32_t i = 0; i < num_rows; ++i) {
          col_base[i] = row_base[row_offsets[i]];
        }
        break;
      case 2:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint16_t*>(col_base)[i] =
              *reinterpret_cast<const uint16_t*>(row_base + row_offsets[i]);
        }
        break;
      case 4:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint32_t*>(col_base)[i] =
              *reinterpret_cast<const uint32_t*>(row_base + row_offsets[i]);
        }
        break;
      case 8:
        for (uint32_t i = 0; i < num_rows; ++i) {
          reinterpret_cast<uint64_t*>(col_base)[i] =
              *reinterpret_cast<const uint64_t*>(row_base + row_offsets[i]);
        }
        break;
      default:
        DCHECK(false);
    }
  }

  if (UsesTransform(*col)) {
    PostDecode(col_prep, col, ctx);
  }
}

bool EncoderBinary::IsInteger(const KeyColumnMetadata& metadata) {
  if (metadata.is_null_type) {
    return false;
  }
  bool is_fixed_length = metadata.is_fixed_length;
  auto size = metadata.fixed_length;
  return is_fixed_length &&
         (size == 0 || size == 1 || size == 2 || size == 4 || size == 8);
}

void EncoderBinary::Decode(uint32_t start_row, uint32_t num_rows,
                                       uint32_t offset_within_row, const RowTable& rows,
                                       KeyColumnArray* col, LightContext* ctx,
                                       KeyColumnArray* temp) {
  if (IsInteger(col->metadata())) {
    EncoderInteger::Decode(start_row, num_rows, offset_within_row, rows, col, ctx, temp);
  } else {
    KeyColumnArray col_prep;
    if (EncoderInteger::UsesTransform(*col)) {
      col_prep = EncoderInteger::ArrayReplace(*col, *temp);
    } else {
      col_prep = *col;
    }

    bool is_row_fixed_length = rows.metadata().is_fixed_length;

#if defined(ARROW_HAVE_AVX2)
    if (ctx->has_avx2()) {
      DecodeHelper_avx2(is_row_fixed_length, start_row, num_rows, offset_within_row, rows,
                        col);
    } else {
#endif
      if (is_row_fixed_length) {
        DecodeImp<true>(start_row, num_rows, offset_within_row, rows, col);
      } else {
        DecodeImp<false>(start_row, num_rows, offset_within_row, rows, col);
      }
#if defined(ARROW_HAVE_AVX2)
    }
#endif

    if (EncoderInteger::UsesTransform(*col)) {
      EncoderInteger::PostDecode(col_prep, col, ctx);
    }
  }
}

template <bool is_row_fixed_length>
void EncoderBinary::DecodeImp(uint32_t start_row, uint32_t num_rows,
                                          uint32_t offset_within_row,
                                          const RowTable& rows, KeyColumnArray* col) {
  DecodeHelper<is_row_fixed_length>(
      start_row, num_rows, offset_within_row, &rows, nullptr, col, col,
      [](uint8_t* dst, const uint8_t* src, int64_t length) {
        for (uint32_t istripe = 0; istripe < bit_util::CeilDiv(length, 8); ++istripe) {
          auto dst64 = reinterpret_cast<uint64_t*>(dst);
          auto src64 = reinterpret_cast<const uint64_t*>(src);
          util::SafeStore(dst64 + istripe, src64[istripe]);
        }
      });
}

void EncoderBinaryPair::Decode(uint32_t start_row, uint32_t num_rows,
                                           uint32_t offset_within_row,
                                           const RowTable& rows, KeyColumnArray* col1,
                                           KeyColumnArray* col2, LightContext* ctx,
                                           KeyColumnArray* temp1, KeyColumnArray* temp2) {
  DCHECK(CanProcessPair(col1->metadata(), col2->metadata()));

  KeyColumnArray col_prep[2];
  if (EncoderInteger::UsesTransform(*col1)) {
    col_prep[0] = EncoderInteger::ArrayReplace(*col1, *temp1);
  } else {
    col_prep[0] = *col1;
  }
  if (EncoderInteger::UsesTransform(*col2)) {
    col_prep[1] = EncoderInteger::ArrayReplace(*col2, *temp2);
  } else {
    col_prep[1] = *col2;
  }

  uint32_t col_width1 = col_prep[0].metadata().fixed_length;
  uint32_t col_width2 = col_prep[1].metadata().fixed_length;
  int log_col_width1 =
      col_width1 == 8 ? 3 : col_width1 == 4 ? 2 : col_width1 == 2 ? 1 : 0;
  int log_col_width2 =
      col_width2 == 8 ? 3 : col_width2 == 4 ? 2 : col_width2 == 2 ? 1 : 0;

  bool is_row_fixed_length = rows.metadata().is_fixed_length;

  uint32_t num_processed = 0;
#if defined(ARROW_HAVE_AVX2)
  if (ctx->has_avx2() && col_width1 == col_width2) {
    num_processed =
        DecodeHelper_avx2(is_row_fixed_length, col_width1, start_row, num_rows,
                          offset_within_row, rows, &col_prep[0], &col_prep[1]);
  }
#endif
  if (num_processed < num_rows) {
    using DecodeImp_t = void (*)(uint32_t, uint32_t, uint32_t, uint32_t, const RowTable&,
                                 KeyColumnArray*, KeyColumnArray*);
    static const DecodeImp_t DecodeImp_fn[] = {
        DecodeImp<false, uint8_t, uint8_t>,   DecodeImp<false, uint16_t, uint8_t>,
        DecodeImp<false, uint32_t, uint8_t>,  DecodeImp<false, uint64_t, uint8_t>,
        DecodeImp<false, uint8_t, uint16_t>,  DecodeImp<false, uint16_t, uint16_t>,
        DecodeImp<false, uint32_t, uint16_t>, DecodeImp<false, uint64_t, uint16_t>,
        DecodeImp<false, uint8_t, uint32_t>,  DecodeImp<false, uint16_t, uint32_t>,
        DecodeImp<false, uint32_t, uint32_t>, DecodeImp<false, uint64_t, uint32_t>,
        DecodeImp<false, uint8_t, uint64_t>,  DecodeImp<false, uint16_t, uint64_t>,
        DecodeImp<false, uint32_t, uint64_t>, DecodeImp<false, uint64_t, uint64_t>,
        DecodeImp<true, uint8_t, uint8_t>,    DecodeImp<true, uint16_t, uint8_t>,
        DecodeImp<true, uint32_t, uint8_t>,   DecodeImp<true, uint64_t, uint8_t>,
        DecodeImp<true, uint8_t, uint16_t>,   DecodeImp<true, uint16_t, uint16_t>,
        DecodeImp<true, uint32_t, uint16_t>,  DecodeImp<true, uint64_t, uint16_t>,
        DecodeImp<true, uint8_t, uint32_t>,   DecodeImp<true, uint16_t, uint32_t>,
        DecodeImp<true, uint32_t, uint32_t>,  DecodeImp<true, uint64_t, uint32_t>,
        DecodeImp<true, uint8_t, uint64_t>,   DecodeImp<true, uint16_t, uint64_t>,
        DecodeImp<true, uint32_t, uint64_t>,  DecodeImp<true, uint64_t, uint64_t>};
    int dispatch_const =
        (log_col_width2 << 2) | log_col_width1 | (is_row_fixed_length ? 16 : 0);
    DecodeImp_fn[dispatch_const](num_processed, start_row, num_rows, offset_within_row,
                                 rows, &(col_prep[0]), &(col_prep[1]));
  }

  if (EncoderInteger::UsesTransform(*col1)) {
    EncoderInteger::PostDecode(col_prep[0], col1, ctx);
  }
  if (EncoderInteger::UsesTransform(*col2)) {
    EncoderInteger::PostDecode(col_prep[1], col2, ctx);
  }
}

template <bool is_row_fixed_length, typename col1_type, typename col2_type>
void EncoderBinaryPair::DecodeImp(uint32_t num_rows_to_skip,
                                              uint32_t start_row, uint32_t num_rows,
                                              uint32_t offset_within_row,
                                              const RowTable& rows, KeyColumnArray* col1,
                                              KeyColumnArray* col2) {
  DCHECK(rows.length() >= start_row + num_rows);
  DCHECK(col1->length() == num_rows && col2->length() == num_rows);

  uint8_t* dst_A = col1->mutable_data(1);
  uint8_t* dst_B = col2->mutable_data(1);

  uint32_t fixed_length = rows.metadata().fixed_length;
  const uint32_t* offsets;
  const uint8_t* src_base;
  if (is_row_fixed_length) {
    src_base = rows.data(1) + fixed_length * start_row + offset_within_row;
    offsets = nullptr;
  } else {
    src_base = rows.data(2) + offset_within_row;
    offsets = rows.offsets() + start_row;
  }

  using col1_type_const = typename std::add_const<col1_type>::type;
  using col2_type_const = typename std::add_const<col2_type>::type;

  if (is_row_fixed_length) {
    const uint8_t* src = src_base + num_rows_to_skip * fixed_length;
    for (uint32_t i = num_rows_to_skip; i < num_rows; ++i) {
      reinterpret_cast<col1_type*>(dst_A)[i] = *reinterpret_cast<col1_type_const*>(src);
      reinterpret_cast<col2_type*>(dst_B)[i] =
          *reinterpret_cast<col2_type_const*>(src + sizeof(col1_type));
      src += fixed_length;
    }
  } else {
    for (uint32_t i = num_rows_to_skip; i < num_rows; ++i) {
      const uint8_t* src = src_base + offsets[i];
      reinterpret_cast<col1_type*>(dst_A)[i] = *reinterpret_cast<col1_type_const*>(src);
      reinterpret_cast<col2_type*>(dst_B)[i] =
          *reinterpret_cast<col2_type_const*>(src + sizeof(col1_type));
    }
  }
}

void EncoderOffsets::Decode(
    uint32_t start_row, uint32_t num_rows, const RowTable& rows,
    std::vector<KeyColumnArray>* varbinary_cols,
    const std::vector<uint32_t>& varbinary_cols_base_offset, LightContext* ctx) {
  DCHECK(!varbinary_cols->empty());
  DCHECK(varbinary_cols->size() == varbinary_cols_base_offset.size());

  DCHECK(!rows.metadata().is_fixed_length);
  DCHECK(rows.length() >= start_row + num_rows);
  for (const auto& col : *varbinary_cols) {
    // Rows and columns must all be varying-length
    DCHECK(!col.metadata().is_fixed_length);
    // The space in columns must be exactly equal to a subset of rows selected
    DCHECK(col.length() == num_rows);
  }

  // Offsets of varbinary columns data within each encoded row are stored
  // in the same encoded row as an array of 32-bit integers.
  // This array follows immediately the data of fixed-length columns.
  // There is one element for each varying-length column.
  // The Nth element is the sum of all the lengths of varbinary columns data in
  // that row, up to and including Nth varbinary column.

  const uint32_t* row_offsets = rows.offsets() + start_row;

  // Set the base offset for each column
  for (size_t col = 0; col < varbinary_cols->size(); ++col) {
    uint32_t* col_offsets = (*varbinary_cols)[col].mutable_offsets();
    col_offsets[0] = varbinary_cols_base_offset[col];
  }

  int string_alignment = rows.metadata().string_alignment;

  for (uint32_t i = 0; i < num_rows; ++i) {
    // Find the beginning of cumulative lengths array for next row
    const uint8_t* row = rows.data(2) + row_offsets[i];
    const uint32_t* varbinary_ends = rows.metadata().varbinary_end_array(row);

    // Update the offset of each column
    uint32_t offset_within_row = rows.metadata().fixed_length;
    for (size_t col = 0; col < varbinary_cols->size(); ++col) {
      offset_within_row +=
          RowTableMetadata::padding_for_alignment(offset_within_row, string_alignment);
      uint32_t length = varbinary_ends[col] - offset_within_row;
      offset_within_row = varbinary_ends[col];
      uint32_t* col_offsets = (*varbinary_cols)[col].mutable_offsets();
      col_offsets[i + 1] = col_offsets[i] + length;
    }
  }
}

void EncoderVarBinary::Decode(uint32_t start_row, uint32_t num_rows,
                                          uint32_t varbinary_col_id, const RowTable& rows,
                                          KeyColumnArray* col, LightContext* ctx) {
  // Output column varbinary buffer needs an extra 32B
  // at the end in avx2 version and 8B otherwise.
#if defined(ARROW_HAVE_AVX2)
  if (ctx->has_avx2()) {
    DecodeHelper_avx2(start_row, num_rows, varbinary_col_id, rows, col);
  } else {
#endif
    if (varbinary_col_id == 0) {
      DecodeImp<true>(start_row, num_rows, varbinary_col_id, rows, col);
    } else {
      DecodeImp<false>(start_row, num_rows, varbinary_col_id, rows, col);
    }
#if defined(ARROW_HAVE_AVX2)
  }
#endif
}

template <bool first_varbinary_col>
void EncoderVarBinary::DecodeImp(uint32_t start_row, uint32_t num_rows,
                                             uint32_t varbinary_col_id,
                                             const RowTable& rows, KeyColumnArray* col) {
  DecodeHelper<first_varbinary_col>(
      start_row, num_rows, varbinary_col_id, &rows, nullptr, col, col,
      [](uint8_t* dst, const uint8_t* src, int64_t length) {
        for (uint32_t istripe = 0; istripe < bit_util::CeilDiv(length, 8); ++istripe) {
          auto dst64 = reinterpret_cast<uint64_t*>(dst);
          auto src64 = reinterpret_cast<const uint64_t*>(src);
          util::SafeStore(dst64 + istripe, src64[istripe]);
        }
      });
}

void EncoderNulls::Decode(uint32_t start_row, uint32_t num_rows,
                                      const RowTable& rows,
                                      std::vector<KeyColumnArray>* cols) {
  // Every output column needs to have a space for exactly the required number
  // of rows. It also needs to have non-nulls bit-vector allocated and mutable.
  DCHECK_GT(cols->size(), 0);
  for (auto& col : *cols) {
    DCHECK(col.length() == num_rows);
    DCHECK(col.mutable_data(0) || col.metadata().is_null_type);
  }

  const uint8_t* null_masks = rows.null_masks();
  uint32_t null_masks_bytes_per_row = rows.metadata().null_masks_bytes_per_row;
  for (size_t col = 0; col < cols->size(); ++col) {
    if ((*cols)[col].metadata().is_null_type) {
      continue;
    }
    uint8_t* non_nulls = (*cols)[col].mutable_data(0);
    const int bit_offset = (*cols)[col].bit_offset(0);
    DCHECK_LT(bit_offset, 8);
    non_nulls[0] |= 0xff << (bit_offset);
    if (bit_offset + num_rows > 8) {
      int bits_in_first_byte = 8 - bit_offset;
      memset(non_nulls + 1, 0xff, bit_util::BytesForBits(num_rows - bits_in_first_byte));
    }
    for (uint32_t row = 0; row < num_rows; ++row) {
      uint32_t null_masks_bit_id =
          (start_row + row) * null_masks_bytes_per_row * 8 + static_cast<uint32_t>(col);
      bool is_set = bit_util::GetBit(null_masks, null_masks_bit_id);
      if (is_set) {
        bit_util::ClearBit(non_nulls, bit_offset + row);
      }
    }
  }
}


template <bool is_row_fixed_length, class COPY_FN>
inline void EncoderBinary::DecodeHelper(
    uint32_t start_row, uint32_t num_rows, uint32_t offset_within_row,
    const RowTable* rows_const, RowTable* rows_mutable_maybe_null,
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
inline void EncoderVarBinary::DecodeHelper(
    uint32_t start_row, uint32_t num_rows, uint32_t varbinary_col_id,
    const RowTable* rows_const, RowTable* rows_mutable_maybe_null,
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

}  // namespace compute
}  // namespace arrow
