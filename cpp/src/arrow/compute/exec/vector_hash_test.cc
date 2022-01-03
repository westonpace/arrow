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

#include <gmock/gmock-matchers.h>

#include <chrono>
#include <map>
#include <set>
#include "arrow/compute/exec/key_hash.h"
#include "arrow/compute/exec/test_util.h"
#include "arrow/compute/exec/util.h"
#include "arrow/util/cpu_info.h"

namespace arrow {
namespace compute {

void TestVectorHashImp(Random64Bit& random, bool use_32bit_hash, bool use_varlen_input,
                       int min_length, int max_length) {
  ARROW_DCHECK(use_varlen_input || min_length == max_length);

  constexpr int min_num_unique = 100;
  constexpr int max_num_unique = 1000;
  constexpr int min_num_rows = 4000;
  constexpr int max_num_rows = 64000;
  int num_unique =
      min_num_unique + (random.next() % (max_num_unique - min_num_unique + 1));
  int num_rows = min_num_rows + (random.next() % (max_num_rows - min_num_rows + 1));

  SCOPED_TRACE("num_bits = " + std::to_string(use_32bit_hash ? 32 : 64) +
               " varlen = " + std::string(use_varlen_input ? "yes" : "no") +
               " num_unique " + std::to_string(num_unique) + " num_rows " +
               std::to_string(num_rows) + " min_length " + std::to_string(min_length) +
               " max_length " + std::to_string(max_length));

  if (max_length == 1) {
    num_unique &= 0x7f;
  }

  std::vector<uint32_t> unique_keys_offsets;
  unique_keys_offsets.resize(num_unique + 1);
  unique_keys_offsets[0] = 0;

  const int num_bytes = unique_keys_offsets[num_unique];
  std::vector<uint8_t> unique_keys;
  unique_keys.resize(num_bytes);
  std::set<std::string> unique_key_strings;
  for (int i = 0; i < num_unique; ++i) {
    for (;;) {
      int next_length;
      if (use_varlen_input) {
        next_length = min_length + random.next() % (max_length - min_length + 1);
      } else {
        next_length = max_length;
      }
      unique_keys_offsets[i + 1] = unique_keys_offsets[i] + next_length;
      unique_keys.resize(unique_keys_offsets[i + 1]);
      uint8_t* next_key = unique_keys.data() + unique_keys_offsets[i];

      for (int iword = 0; iword < next_length / static_cast<int>(sizeof(uint64_t));
           ++iword) {
        reinterpret_cast<uint64_t*>(next_key)[iword] = random.next();
      }
      if (next_length % sizeof(uint64_t) > 0) {
        uint8_t* tail = next_key + next_length - (next_length % sizeof(uint64_t));
        for (int ibyte = 0; ibyte < (next_length % static_cast<int>(sizeof(uint64_t)));
             ++ibyte) {
          tail[ibyte] = static_cast<uint8_t>(random.next() & 0xff);
        }
      }
      std::string next_key_string =
          std::string(reinterpret_cast<const char*>(next_key), next_length);
      if (unique_key_strings.find(next_key_string) == unique_key_strings.end()) {
        unique_key_strings.insert(next_key_string);
        break;
      }
    }
  }

  std::vector<int> row_ids;
  row_ids.resize(num_rows);
  std::vector<uint8_t> keys;
  std::vector<uint32_t> keys_offsets;
  keys_offsets.resize(num_rows + 1);
  keys_offsets[0] = 0;
  for (int i = 0; i < num_rows; ++i) {
    int row_id = random.next() % num_unique;
    row_ids[i] = row_id;
    int next_length = unique_keys_offsets[row_id + 1] - unique_keys_offsets[row_id];
    keys_offsets[i + 1] = keys_offsets[i] + next_length;
  }
  keys.resize(keys_offsets[num_rows]);
  for (int i = 0; i < num_rows; ++i) {
    int row_id = row_ids[i];
    int next_length = keys_offsets[i + 1] - keys_offsets[i];
    memcpy(keys.data() + keys_offsets[i],
           unique_keys.data() + unique_keys_offsets[row_id], next_length);
  }

  constexpr int min_rows_for_timing = 1 << 23;
  int num_repeats = static_cast<int>(bit_util::CeilDiv(min_rows_for_timing, num_rows));
#ifndef NDEBUG
  num_repeats = 1;
#endif

  std::vector<uint32_t> hashes_scalar32;
  std::vector<uint64_t> hashes_scalar64;
  hashes_scalar32.resize(num_rows);
  hashes_scalar64.resize(num_rows);
  std::vector<uint32_t> hashes_simd32;
  std::vector<uint64_t> hashes_simd64;
  hashes_simd32.resize(num_rows);
  hashes_simd64.resize(num_rows);

  int64_t hardware_flags_scalar = 0LL;
  int64_t hardware_flags_simd = ::arrow::internal::CpuInfo::AVX2;

  constexpr int mini_batch_size = 1024;
  std::vector<uint32_t> temp_buffer;
  temp_buffer.resize(mini_batch_size * 4);

  for (bool use_simd : {false, true}) {
    for (int i = 0; i < num_repeats; ++i) {
      if (use_32bit_hash) {
        if (!use_varlen_input) {
          Hashing32::hash_fixed(use_simd ? hardware_flags_simd : hardware_flags_scalar,
                                /*combine_hashes=*/false, num_rows, max_length,
                                keys.data(),
                                use_simd ? hashes_simd32.data() : hashes_scalar32.data(),
                                temp_buffer.data());
        } else {
          for (int first_row = 0; first_row < num_rows;) {
            int batch_size_next = std::min(num_rows - first_row, mini_batch_size);

            Hashing32::hash_varlen(
                use_simd ? hardware_flags_simd : hardware_flags_scalar,
                /*combine_hashes=*/false, batch_size_next,
                keys_offsets.data() + first_row, keys.data(),
                (use_simd ? hashes_simd32.data() : hashes_scalar32.data()) + first_row,
                temp_buffer.data());

            first_row += batch_size_next;
          }
        }
      } else {
        if (!use_varlen_input) {
          Hashing64::hash_fixed(/*combine_hashes=*/false, num_rows, max_length,
                                keys.data(),
                                use_simd ? hashes_simd64.data() : hashes_scalar64.data());
        } else {
          Hashing64::hash_varlen(
              /*combine_hashes=*/false, num_rows, keys_offsets.data(), keys.data(),
              use_simd ? hashes_simd64.data() : hashes_scalar64.data());
        }
      }
    }
  }
  if (use_32bit_hash) {
    for (int i = 0; i < num_rows; ++i) {
      hashes_scalar64[i] = hashes_scalar32[i];
      hashes_simd64[i] = hashes_simd32[i];
    }
  }

  // Verify that both scalar and AVX2 implementations give the same hashes
  //
  for (int i = 0; i < num_rows; ++i) {
    if (hashes_scalar64[i] != hashes_simd64[i]) {
      ARROW_DCHECK(false);
    }
  }

  // Verify that the same key appearing multiple times generates the same hash
  // each time. Measure the number of unique hashes and compare to the number
  // of unique keys.
  //
  std::map<int, uint64_t> unique_key_to_hash;
  std::set<uint64_t> unique_hashes;
  for (int i = 0; i < num_rows; ++i) {
    std::map<int, uint64_t>::iterator iter = unique_key_to_hash.find(row_ids[i]);
    if (iter == unique_key_to_hash.end()) {
      unique_key_to_hash.insert(std::make_pair(row_ids[i], hashes_scalar64[i]));
    } else {
      ARROW_DCHECK(iter->second == hashes_scalar64[i]);
    }
    if (unique_hashes.find(hashes_scalar64[i]) == unique_hashes.end()) {
      unique_hashes.insert(hashes_scalar64[i]);
    }
  }
  float percent_hash_collisions = 100.0f *
                                  static_cast<float>(num_unique - unique_hashes.size()) /
                                  static_cast<float>(num_unique);
  SCOPED_TRACE("percent_hash_collisions " + std::to_string(percent_hash_collisions));
  ARROW_DCHECK(percent_hash_collisions < 5.0f);
}

TEST(VectorHash, Basic) {
  Random64Bit random(/*seed=*/0);

  int numtest = 100;

  constexpr int min_length = 1;
  constexpr int max_length = 50;

  for (bool use_32bit_hash : {true, false}) {
    for (bool use_varlen_input : {false, true}) {
      for (int itest = 0; itest < numtest; ++itest) {
        int length = static_cast<int>(std::max(
            static_cast<uint64_t>(use_varlen_input ? 2 : 1),
            static_cast<uint64_t>(min_length +
                                  random.next() % (max_length - min_length + 1))));

        TestVectorHashImp(random, use_32bit_hash, use_varlen_input,
                          use_varlen_input ? 0 : length, length);
      }
    }
  }
}

}  // namespace compute
}  // namespace arrow
