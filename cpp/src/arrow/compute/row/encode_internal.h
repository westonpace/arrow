#include "arrow/compute/row/encode.h"
#include "arrow/compute/row/row_internal.h"
#include "arrow/compute/light_array.h"

namespace arrow {
namespace compute {

class EncoderInteger {
 public:
  static void Decode(uint32_t start_row, uint32_t num_rows, uint32_t offset_within_row,
                     const RowTable& rows, KeyColumnArray* col, LightContext* ctx,
                     KeyColumnArray* temp);
  static bool UsesTransform(const KeyColumnArray& column);
  static KeyColumnArray ArrayReplace(const KeyColumnArray& column,
                                     const KeyColumnArray& temp);
  static void PostDecode(const KeyColumnArray& input, KeyColumnArray* output,
                         LightContext* ctx);

 private:
  static bool IsBoolean(const KeyColumnMetadata& metadata);
};

class EncoderBinary {
 public:
  static void EncodeSelected(uint32_t offset_within_row, RowTable* rows,
                             const KeyColumnArray& col, uint32_t num_selected,
                             const uint16_t* selection);
  static void Decode(uint32_t start_row, uint32_t num_rows, uint32_t offset_within_row,
                     const RowTable& rows, KeyColumnArray* col, LightContext* ctx,
                     KeyColumnArray* temp);
  static bool IsInteger(const KeyColumnMetadata& metadata);

 private:
  template <class COPY_FN, class SET_NULL_FN>
  static void EncodeSelectedImp(uint32_t offset_within_row, RowTable* rows,
                                const KeyColumnArray& col, uint32_t num_selected,
                                const uint16_t* selection, COPY_FN copy_fn,
                                SET_NULL_FN set_null_fn);

  template <bool is_row_fixed_length, class COPY_FN>
  static inline void DecodeHelper(uint32_t start_row, uint32_t num_rows,
                                  uint32_t offset_within_row, const RowTable* rows_const,
                                  RowTable* rows_mutable_maybe_null,
                                  const KeyColumnArray* col_const,
                                  KeyColumnArray* col_mutable_maybe_null,
                                  COPY_FN copy_fn);
  template <bool is_row_fixed_length>
  static void DecodeImp(uint32_t start_row, uint32_t num_rows, uint32_t offset_within_row,
                        const RowTable& rows, KeyColumnArray* col);
#if defined(ARROW_HAVE_AVX2)
  static void DecodeHelper_avx2(bool is_row_fixed_length, uint32_t start_row,
                                uint32_t num_rows, uint32_t offset_within_row,
                                const RowTable& rows, KeyColumnArray* col);
  template <bool is_row_fixed_length>
  static void DecodeImp_avx2(uint32_t start_row, uint32_t num_rows,
                             uint32_t offset_within_row, const RowTable& rows,
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
                     const RowTable& rows, KeyColumnArray* col1, KeyColumnArray* col2,
                     LightContext* ctx, KeyColumnArray* temp1,
                     KeyColumnArray* temp2);

 private:
  template <bool is_row_fixed_length, typename col1_type, typename col2_type>
  static void DecodeImp(uint32_t num_rows_to_skip, uint32_t start_row, uint32_t num_rows,
                        uint32_t offset_within_row, const RowTable& rows,
                        KeyColumnArray* col1, KeyColumnArray* col2);
#if defined(ARROW_HAVE_AVX2)
  static uint32_t DecodeHelper_avx2(bool is_row_fixed_length, uint32_t col_width,
                                    uint32_t start_row, uint32_t num_rows,
                                    uint32_t offset_within_row, const RowTable& rows,
                                    KeyColumnArray* col1, KeyColumnArray* col2);
  template <bool is_row_fixed_length, uint32_t col_width>
  static uint32_t DecodeImp_avx2(uint32_t start_row, uint32_t num_rows,
                                 uint32_t offset_within_row, const RowTable& rows,
                                 KeyColumnArray* col1, KeyColumnArray* col2);
#endif
};

class EncoderOffsets {
 public:
  static void GetRowOffsetsSelected(RowTable* rows,
                                    const std::vector<KeyColumnArray>& cols,
                                    uint32_t num_selected, const uint16_t* selection);
  static void EncodeSelected(RowTable* rows, const std::vector<KeyColumnArray>& cols,
                             uint32_t num_selected, const uint16_t* selection);

  static void Decode(uint32_t start_row, uint32_t num_rows, const RowTable& rows,
                     std::vector<KeyColumnArray>* varbinary_cols,
                     const std::vector<uint32_t>& varbinary_cols_base_offset,
                     LightContext* ctx);

 private:
  template <bool has_nulls, bool is_first_varbinary>
  static void EncodeSelectedImp(uint32_t ivarbinary, RowTable* rows,
                                const std::vector<KeyColumnArray>& cols,
                                uint32_t num_selected, const uint16_t* selection);
};

class EncoderVarBinary {
 public:
  static void EncodeSelected(uint32_t ivarbinary, RowTable* rows,
                             const KeyColumnArray& cols, uint32_t num_selected,
                             const uint16_t* selection);

  static void Decode(uint32_t start_row, uint32_t num_rows, uint32_t varbinary_col_id,
                     const RowTable& rows, KeyColumnArray* col, LightContext* ctx);

 private:
  template <bool first_varbinary_col, class COPY_FN>
  static inline void DecodeHelper(uint32_t start_row, uint32_t num_rows,
                                  uint32_t varbinary_col_id, const RowTable* rows_const,
                                  RowTable* rows_mutable_maybe_null,
                                  const KeyColumnArray* col_const,
                                  KeyColumnArray* col_mutable_maybe_null,
                                  COPY_FN copy_fn);
  template <bool first_varbinary_col>
  static void DecodeImp(uint32_t start_row, uint32_t num_rows, uint32_t varbinary_col_id,
                        const RowTable& rows, KeyColumnArray* col);
#if defined(ARROW_HAVE_AVX2)
  static void DecodeHelper_avx2(uint32_t start_row, uint32_t num_rows,
                                uint32_t varbinary_col_id, const RowTable& rows,
                                KeyColumnArray* col);
  template <bool first_varbinary_col>
  static void DecodeImp_avx2(uint32_t start_row, uint32_t num_rows,
                             uint32_t varbinary_col_id, const RowTable& rows,
                             KeyColumnArray* col);
#endif
};

class EncoderNulls {
 public:
  static void EncodeSelected(RowTable* rows, const std::vector<KeyColumnArray>& cols,
                             uint32_t num_selected, const uint16_t* selection);

  static void Decode(uint32_t start_row, uint32_t num_rows, const RowTable& rows,
                     std::vector<KeyColumnArray>* cols);
};

}  // namespace compute
}  // namespace arrow
