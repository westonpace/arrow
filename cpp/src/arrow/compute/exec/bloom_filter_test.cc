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

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <set>
#include <thread>
#include "arrow/compute/exec/bloom_filter.h"
#include "arrow/compute/exec/key_hash.h"
#include "arrow/compute/exec/test_util.h"
#include "arrow/compute/exec/util.h"
#include "arrow/util/bitmap_ops.h"
#include "arrow/util/cpu_info.h"

namespace arrow {
namespace compute {

Status BuildBloomFilter(BloomFilterBuildStrategy strategy, size_t num_threads,
                        int64_t hardware_flags, MemoryPool* pool, int64_t num_rows,
                        std::function<void(int64_t, int, uint32_t*)> get_hash32_impl,
                        std::function<void(int64_t, int, uint64_t*)> get_hash64_impl,
                        BlockedBloomFilter* target, float* build_cost) {
  constexpr int batch_size_max = 32 * 1024;
  int64_t num_batches = bit_util::CeilDiv(num_rows, batch_size_max);

  auto builder = BloomFilterBuilder::Make(strategy);

  std::vector<std::vector<uint32_t>> thread_local_hashes32;
  std::vector<std::vector<uint64_t>> thread_local_hashes64;
  thread_local_hashes32.resize(num_threads);
  thread_local_hashes64.resize(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    thread_local_hashes32[i].resize(batch_size_max);
    thread_local_hashes64[i].resize(batch_size_max);
  }

  // Repeate the entire test in a loop multiple times in order to get meaningful time
  // measurements. Time measurements in debug do not provide useful information and would
  // make test take unreasonably long, so there are no repeats in debug.
  //
  std::vector<float> build_cost_vector;
  int64_t num_repeats =
      std::max(static_cast<int64_t>(1), bit_util::CeilDiv(1LL << 27, num_rows));
#ifndef NDEBUG
  num_repeats = 1LL;
#endif
  build_cost_vector.resize(num_repeats);

  for (int64_t irepeat = 0; irepeat < num_repeats; ++irepeat) {
    auto time0 = std::chrono::high_resolution_clock::now();

    RETURN_NOT_OK(builder->Begin(num_threads, hardware_flags, pool, num_rows,
                                 bit_util::CeilDiv(num_rows, batch_size_max), target));

    for (int64_t i = 0; i < num_batches; ++i) {
      size_t thread_index = 0;
      int batch_size = static_cast<int>(
          std::min(num_rows - i * batch_size_max, static_cast<int64_t>(batch_size_max)));
      if (target->NumHashBitsUsed() > 32) {
        uint64_t* hashes = thread_local_hashes64[thread_index].data();
        get_hash64_impl(i * batch_size_max, batch_size, hashes);
        Status status = builder->PushNextBatch(thread_index, batch_size, hashes);
        ARROW_DCHECK(status.ok());
      } else {
        uint32_t* hashes = thread_local_hashes32[thread_index].data();
        get_hash32_impl(i * batch_size_max, batch_size, hashes);
        Status status = builder->PushNextBatch(thread_index, batch_size, hashes);
        ARROW_DCHECK(status.ok());
      }
    }

    auto time1 = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time1 - time0).count();

    builder->CleanUp();

    build_cost_vector[irepeat] = static_cast<float>(ns) / static_cast<float>(num_rows);
  }

  std::sort(build_cost_vector.begin(), build_cost_vector.end());
  *build_cost = build_cost_vector[build_cost_vector.size() / 2];

  return Status::OK();
}

// FPR (false positives rate) - fraction of false positives relative to the sum
// of false positives and true negatives.
//
// Output FPR and build and probe cost.
//
Status TestBloomSmall(BloomFilterBuildStrategy strategy, int64_t num_build,
                      int num_build_copies, int dop, bool use_simd,
                      bool enable_prefetch) {
  int64_t hardware_flags = use_simd ? ::arrow::internal::CpuInfo::AVX2 : 0;

  // Generate input keys
  //
  int64_t num_probe = 4 * num_build;
  Random64Bit rnd(/*seed=*/0);
  std::vector<uint64_t> unique_keys;
  {
    std::set<uint64_t> unique_keys_set;
    for (int64_t i = 0; i < num_build + num_probe; ++i) {
      uint64_t value;
      for (;;) {
        value = rnd.next();
        if (unique_keys_set.find(value) == unique_keys_set.end()) {
          break;
        }
      }
      unique_keys.push_back(value);
      unique_keys_set.insert(value);
    }
  }

  // Generate input hashes
  //
  std::vector<uint32_t> hashes32;
  std::vector<uint64_t> hashes64;
  hashes32.resize(unique_keys.size());
  hashes64.resize(unique_keys.size());
  int batch_size_max = 1024;
  for (size_t i = 0; i < unique_keys.size(); i += batch_size_max) {
    int batch_size = static_cast<int>(
        std::min(unique_keys.size() - i, static_cast<size_t>(batch_size_max)));
    constexpr int key_length = sizeof(uint64_t);
    Hashing32::hash_fixed(hardware_flags, /*combine_hashes=*/false, batch_size,
                          key_length,
                          reinterpret_cast<const uint8_t*>(unique_keys.data() + i),
                          hashes32.data() + i, nullptr);
    Hashing64::hash_fixed(
        /*combine_hashes=*/false, batch_size, key_length,
        reinterpret_cast<const uint8_t*>(unique_keys.data() + i), hashes64.data() + i);
  }

  MemoryPool* pool = default_memory_pool();

  // Build the filter
  //
  BlockedBloomFilter reference;
  BlockedBloomFilter bloom;
  float build_cost_single_threaded;
  float build_cost;

  RETURN_NOT_OK(BuildBloomFilter(
      BloomFilterBuildStrategy::SINGLE_THREADED, dop, hardware_flags, pool, num_build,
      [hashes32](int64_t first_row, int num_rows, uint32_t* output_hashes) {
        memcpy(output_hashes, hashes32.data() + first_row, num_rows * sizeof(uint32_t));
      },
      [hashes64](int64_t first_row, int num_rows, uint64_t* output_hashes) {
        memcpy(output_hashes, hashes64.data() + first_row, num_rows * sizeof(uint64_t));
      },
      &reference, &build_cost_single_threaded));

  RETURN_NOT_OK(BuildBloomFilter(
      strategy, dop, hardware_flags, pool, num_build * num_build_copies,
      [hashes32, num_build](int64_t first_row, int num_rows, uint32_t* output_hashes) {
        int64_t first_row_clamped = first_row % num_build;
        int64_t num_rows_processed = 0;
        while (num_rows_processed < num_rows) {
          int64_t num_rows_next =
              std::min(static_cast<int64_t>(num_rows) - num_rows_processed,
                       num_build - first_row_clamped);
          memcpy(output_hashes + num_rows_processed, hashes32.data() + first_row_clamped,
                 num_rows_next * sizeof(uint32_t));
          first_row_clamped = 0;
          num_rows_processed += num_rows_next;
        }
      },
      [hashes64, num_build](int64_t first_row, int num_rows, uint64_t* output_hashes) {
        int64_t first_row_clamped = first_row % num_build;
        int64_t num_rows_processed = 0;
        while (num_rows_processed < num_rows) {
          int64_t num_rows_next =
              std::min(static_cast<int64_t>(num_rows) - num_rows_processed,
                       num_build - first_row_clamped);
          memcpy(output_hashes + num_rows_processed, hashes64.data() + first_row_clamped,
                 num_rows_next * sizeof(uint64_t));
          first_row_clamped = 0;
          num_rows_processed += num_rows_next;
        }
      },
      &bloom, &build_cost));

  int log_before = bloom.log_num_blocks();

  if (num_build_copies > 1) {
    reference.Fold();
    bloom.Fold();
  } else {
    if (strategy != BloomFilterBuildStrategy::SINGLE_THREADED) {
      ARROW_DCHECK(reference.IsSameAs(&bloom));
    }
  }

  int log_after = bloom.log_num_blocks();

  float fraction_of_bits_set = static_cast<float>(bloom.NumBitsSet()) /
                               static_cast<float>(64LL << bloom.log_num_blocks());

  ARROW_SCOPED_TRACE("log_before = ", log_before, " log_after = ", log_after,
                     " percent_bits_set = ", 100.0f * fraction_of_bits_set);

  // Verify no false negatives
  //
  for (int64_t i = 0; i < num_build; ++i) {
    bool found;
    if (bloom.NumHashBitsUsed() > 32) {
      found = bloom.Find(hashes64[i]);
    } else {
      found = bloom.Find(hashes32[i]);
    }
    if (!found) {
      ARROW_DCHECK(false);
      break;
    }
  }
  return Status::OK();
}

template <typename T>
void test_Bloom_large_hash(int64_t hardware_flags, int64_t block,
                           const std::vector<uint64_t>& first_in_block, int64_t first_row,
                           int num_rows, T* output_hashes) {
  // Largest 63-bit prime
  constexpr uint64_t prime = 0x7FFFFFFFFFFFFFE7ULL;

  constexpr int mini_batch_size = 1024;
  uint64_t keys[mini_batch_size];
  int64_t ikey = first_row / block * block;
  uint64_t key = first_in_block[first_row / block];
  while (ikey < first_row) {
    key += prime;
    ++ikey;
  }
  for (int ibase = 0; ibase < num_rows;) {
    int next_batch_size = std::min(num_rows - ibase, mini_batch_size);
    for (int i = 0; i < next_batch_size; ++i) {
      keys[i] = key;
      key += prime;
    }

    constexpr int key_length = sizeof(uint64_t);
    if (sizeof(T) == sizeof(uint32_t)) {
      Hashing32::hash_fixed(hardware_flags, false, next_batch_size, key_length,
                            reinterpret_cast<const uint8_t*>(keys),
                            reinterpret_cast<uint32_t*>(output_hashes) + ibase, nullptr);
    } else {
      Hashing64::hash_fixed(false, next_batch_size, key_length,
                            reinterpret_cast<const uint8_t*>(keys),
                            reinterpret_cast<uint64_t*>(output_hashes) + ibase);
    }

    ibase += next_batch_size;
  }
}

// Test with larger size Bloom filters (use large prime with arithmetic
// sequence modulo 2^64).
//
Status TestBloomLarge(BloomFilterBuildStrategy strategy, int64_t num_build, int dop,
                      bool use_simd, bool enable_prefetch) {
  int64_t hardware_flags = use_simd ? ::arrow::internal::CpuInfo::AVX2 : 0;

  // Largest 63-bit prime
  constexpr uint64_t prime = 0x7FFFFFFFFFFFFFE7ULL;

  // Generate input keys
  //
  int64_t num_probe = 4 * num_build;
  const int64_t block = 1024;
  std::vector<uint64_t> first_in_block;
  first_in_block.resize(bit_util::CeilDiv(num_build + num_probe, block));
  uint64_t current = prime;
  for (int64_t i = 0; i < num_build + num_probe; ++i) {
    if (i % block == 0) {
      first_in_block[i / block] = current;
    }
    current += prime;
  }

  MemoryPool* pool = default_memory_pool();

  // Build the filter
  //
  BlockedBloomFilter reference;
  BlockedBloomFilter bloom;
  float build_cost_single_threaded;
  float build_cost;

  for (int ibuild = 0; ibuild < 2; ++ibuild) {
    if (ibuild == 0 && strategy == BloomFilterBuildStrategy::SINGLE_THREADED) {
      continue;
    }
    RETURN_NOT_OK(BuildBloomFilter(
        ibuild == 0 ? BloomFilterBuildStrategy::SINGLE_THREADED : strategy,
        ibuild == 0 ? 1 : dop, hardware_flags, pool, num_build,
        [hardware_flags, &first_in_block](int64_t first_row, int num_rows,
                                          uint32_t* output_hashes) {
          const int64_t block = 1024;
          test_Bloom_large_hash(hardware_flags, block, first_in_block, first_row,
                                num_rows, output_hashes);
        },
        [hardware_flags, &first_in_block](int64_t first_row, int num_rows,
                                          uint64_t* output_hashes) {
          const int64_t block = 1024;
          test_Bloom_large_hash(hardware_flags, block, first_in_block, first_row,
                                num_rows, output_hashes);
        },
        ibuild == 0 ? &reference : &bloom,
        ibuild == 0 ? &build_cost_single_threaded : &build_cost));
  }

  if (strategy != BloomFilterBuildStrategy::SINGLE_THREADED) {
    ARROW_DCHECK(reference.IsSameAs(&bloom));
  }

  std::vector<uint32_t> hashes32;
  std::vector<uint64_t> hashes64;
  std::vector<uint8_t> result_bit_vector;
  hashes32.resize(block);
  hashes64.resize(block);
  result_bit_vector.resize(bit_util::BytesForBits(block));

  int64_t num_repeats = 1LL;
#ifdef NDEBUG
  num_repeats = std::max(1LL, bit_util::CeilDiv(1000000ULL, num_probe));
#endif

  // Verify no false negatives and measure false positives.
  // Measure FPR and performance.
  //
  int64_t num_negatives_build = 0LL;

  for (int64_t i = 0; i < num_build + num_probe * num_repeats;) {
    int64_t first_row = i < num_build ? i : num_build + ((i - num_build) % num_probe);
    int64_t last_row = i < num_build ? num_build : num_build + num_probe;
    int64_t next_batch_size = std::min(last_row - first_row, block);
    if (bloom.NumHashBitsUsed() > 32) {
      test_Bloom_large_hash(hardware_flags, block, first_in_block, first_row,
                            static_cast<int>(next_batch_size), hashes64.data());
      bloom.Find(hardware_flags, next_batch_size, hashes64.data(),
                 result_bit_vector.data(), enable_prefetch);
    } else {
      test_Bloom_large_hash(hardware_flags, block, first_in_block, first_row,
                            static_cast<int>(next_batch_size), hashes32.data());
      bloom.Find(hardware_flags, next_batch_size, hashes32.data(),
                 result_bit_vector.data(), enable_prefetch);
    }
    uint64_t num_negatives = 0ULL;
    for (int iword = 0; iword < next_batch_size / 64; ++iword) {
      uint64_t word = reinterpret_cast<const uint64_t*>(result_bit_vector.data())[iword];
      num_negatives += ARROW_POPCOUNT64(~word);
    }
    if (next_batch_size % 64 > 0) {
      uint64_t word = reinterpret_cast<const uint64_t*>(
          result_bit_vector.data())[next_batch_size / 64];
      uint64_t mask = (1ULL << (next_batch_size % 64)) - 1;
      word |= ~mask;
      num_negatives += ARROW_POPCOUNT64(~word);
    }
    if (i < num_build) {
      num_negatives_build += num_negatives;
    }
    i += next_batch_size;
  }

  ARROW_DCHECK(num_negatives_build == 0);

  return Status::OK();
}

TEST(BloomFilter, Basic) {
  std::vector<int64_t> num_build;
  constexpr int log_min = 8;
  constexpr int log_max = 16;
  constexpr int log_large = 22;
  for (int log_num_build = log_min; log_num_build < log_max; ++log_num_build) {
    constexpr int num_intermediate_points = 2;
    for (int i = 0; i < num_intermediate_points; ++i) {
      int64_t num_left = 1LL << log_num_build;
      int64_t num_right = 1LL << (log_num_build + 1);
      num_build.push_back((num_left * (num_intermediate_points - i) + num_right * i) /
                          num_intermediate_points);
    }
  }
  num_build.push_back(1LL << log_max);
  num_build.push_back(1LL << log_large);

  constexpr int num_param_sets = 3;
  struct {
    bool use_avx2;
    bool enable_prefetch;
    bool insert_multiple_copies;
  } params[num_param_sets];
  for (int i = 0; i < num_param_sets; ++i) {
    params[i].use_avx2 = (i == 1);
    params[i].enable_prefetch = (i == 2);
    params[i].insert_multiple_copies = (i == 3);
  }

  std::vector<BloomFilterBuildStrategy> strategy;
  strategy.push_back(BloomFilterBuildStrategy::SINGLE_THREADED);
  strategy.push_back(BloomFilterBuildStrategy::PARALLEL);

  static constexpr int64_t min_rows_for_large = 2 * 1024 * 1024;

  // Number of parallel threads executing the test
  int dop = 1;

  for (size_t istrategy = 0; istrategy < strategy.size(); ++istrategy) {
    for (int iparam_set = 0; iparam_set < num_param_sets; ++iparam_set) {
      ARROW_SCOPED_TRACE("%s ", params[iparam_set].use_avx2                 ? "AVX2"
                                : params[iparam_set].enable_prefetch        ? "PREFETCH"
                                : params[iparam_set].insert_multiple_copies ? "FOLDING"
                                                                            : "REGULAR");
      for (size_t inum_build = 0; inum_build < num_build.size(); ++inum_build) {
        ARROW_SCOPED_TRACE("num_build ", static_cast<int>(num_build[inum_build]));
        if (num_build[inum_build] >= min_rows_for_large) {
          ASSERT_OK(TestBloomLarge(strategy[istrategy], num_build[inum_build], dop,
                                   params[iparam_set].use_avx2,
                                   params[iparam_set].enable_prefetch));

        } else {
          ASSERT_OK(TestBloomSmall(strategy[istrategy], num_build[inum_build],
                                   params[iparam_set].insert_multiple_copies ? 8 : 1, dop,
                                   params[iparam_set].use_avx2,
                                   params[iparam_set].enable_prefetch));
        }
      }
    }
  }
}

TEST(BloomFilter, Scaling) {
  std::vector<int64_t> num_build;
  num_build.push_back(1000000);
  num_build.push_back(4000000);

  std::vector<int> dop;
  dop.push_back(1);

  std::vector<BloomFilterBuildStrategy> strategy;
  strategy.push_back(BloomFilterBuildStrategy::PARALLEL);

  for (bool use_avx2 : {false, true}) {
    for (size_t istrategy = 0; istrategy < strategy.size(); ++istrategy) {
      for (size_t inum_build = 0; inum_build < num_build.size(); ++inum_build) {
        for (size_t idop = 0; idop < dop.size(); ++idop) {
          ARROW_SCOPED_TRACE("num_build = ", static_cast<int>(num_build[inum_build]));
          ARROW_SCOPED_TRACE("strategy = ",
                             strategy[istrategy] == BloomFilterBuildStrategy::PARALLEL
                                 ? "PARALLEL"
                                 : "SINGLE_THREADED");
          ARROW_SCOPED_TRACE("avx2 = ", use_avx2 ? "AVX2" : "SCALAR");
          ARROW_SCOPED_TRACE("dop = ", dop[idop]);
          ASSERT_OK(TestBloomLarge(strategy[istrategy], num_build[inum_build], dop[idop],
                                   use_avx2,
                                   /*enable_prefetch=*/false));
        }
      }
    }
  }
}

}  // namespace compute
}  // namespace arrow
