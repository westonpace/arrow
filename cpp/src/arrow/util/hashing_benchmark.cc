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

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"

#include "arrow/testing/gtest_util.h"
#include "arrow/util/hashing.h"

namespace arrow {
namespace internal {

template <class Integer>
static std::vector<Integer> MakeIntegers(int32_t n_values) {
  std::vector<Integer> values(n_values);

  std::default_random_engine gen(42);
  std::uniform_int_distribution<Integer> values_dist(0,
                                                     std::numeric_limits<Integer>::max());
  std::generate(values.begin(), values.end(),
                [&]() { return static_cast<Integer>(values_dist(gen)); });
  return values;
}

static std::vector<std::string> MakeStrings(int32_t n_values, int32_t min_length,
                                            int32_t max_length) {
  std::default_random_engine gen(42);
  std::vector<std::string> values(n_values);

  // Generate strings between 2 and 20 bytes
  std::uniform_int_distribution<int32_t> length_dist(min_length, max_length);
  std::independent_bits_engine<std::default_random_engine, 8, uint16_t> bytes_gen(42);

  std::generate(values.begin(), values.end(), [&]() {
    auto length = length_dist(gen);
    std::string s(length, 'X');
    for (int32_t i = 0; i < length; ++i) {
      s[i] = static_cast<uint8_t>(bytes_gen());
    }
    return s;
  });
  return values;
}

static void HashIntegers(benchmark::State& state) {  // NOLINT non-const reference
  const std::vector<int64_t> values = MakeIntegers<int64_t>(10000);

  while (state.KeepRunning()) {
    hash_t total = 0;
    for (const int64_t v : values) {
      total += ScalarHelper<int64_t, 0>::ComputeHash(v);
      total += ScalarHelper<int64_t, 1>::ComputeHash(v);
    }
    benchmark::DoNotOptimize(total);
  }
  state.SetBytesProcessed(2 * state.iterations() * values.size() * sizeof(int64_t));
  state.SetItemsProcessed(2 * state.iterations() * values.size());
}

static void BenchmarkStringHashing(benchmark::State& state,  // NOLINT non-const reference
                                   const std::vector<std::string>& values) {
  uint64_t total_size = 0;
  for (const std::string& v : values) {
    total_size += v.size();
  }

  while (state.KeepRunning()) {
    hash_t total = 0;
    for (const std::string& v : values) {
      total += ComputeStringHash<0>(v.data(), static_cast<int64_t>(v.size()));
      total += ComputeStringHash<1>(v.data(), static_cast<int64_t>(v.size()));
    }
    benchmark::DoNotOptimize(total);
  }
  state.SetBytesProcessed(2 * state.iterations() * total_size);
  state.SetItemsProcessed(2 * state.iterations() * values.size());
}

static void HashSmallStrings(benchmark::State& state) {  // NOLINT non-const reference
  const std::vector<std::string> values = MakeStrings(10000, 2, 20);
  BenchmarkStringHashing(state, values);
}

static void HashMediumStrings(benchmark::State& state) {  // NOLINT non-const reference
  const std::vector<std::string> values = MakeStrings(10000, 20, 120);
  BenchmarkStringHashing(state, values);
}

static void HashLargeStrings(benchmark::State& state) {  // NOLINT non-const reference
  const std::vector<std::string> values = MakeStrings(1000, 120, 2000);
  BenchmarkStringHashing(state, values);
}

void CompareWithMemcmp(const uint8_t* left, const uint8_t* right, uint8_t* out,
                       const int* indices, int length) {
  const int* idx_it = indices;
  const int* idx_end = indices + length;
  uint8_t* out_it = out;
  while (idx_it != idx_end) {
    *out_it = memcmp(left + (*idx_it * 8), right + (*idx_it * 8), 8) == 0;
    out_it++;
    idx_it++;
  }
}

void CompareWithCast(const uint8_t* left, const uint8_t* right, uint8_t* out,
                     const int* indices, int length) {
  const int* idx_it = indices;
  const int* idx_end = indices + length;
  uint8_t* out_it = out;
  while (idx_it != idx_end) {
    *out_it = *(reinterpret_cast<const uint64_t*>(left) + *idx_it) ==
              *(reinterpret_cast<const uint64_t*>(right) + *idx_it);
    out_it++;
    idx_it++;
  }
}

template <bool kUseCast, bool kAligned>
static void RandomCompare(benchmark::State& state) {  // NOLINT non-const reference
  constexpr int kNumElements = 10000;
  const std::vector<int64_t> left = MakeIntegers<int64_t>(kNumElements + 1);
  const std::vector<int64_t> right = MakeIntegers<int64_t>(kNumElements + 1);
  std::vector<int> indices(kNumElements);
  std::vector<uint8_t> matches(kNumElements);
  std::iota(indices.begin(), indices.end(), 0);
  std::default_random_engine gen(42);
  std::shuffle(indices.begin(), indices.end(), gen);
  const uint8_t* left_start = reinterpret_cast<const uint8_t*>(left.data());
  const uint8_t* right_start = reinterpret_cast<const uint8_t*>(right.data());
  if (!kAligned) {
    left_start += 4;
    right_start += 4;
  }
  for (auto _ : state) {
    if (kUseCast) {
      CompareWithCast(left_start, right_start, matches.data(), indices.data(),
                      kNumElements);
    } else {
      CompareWithMemcmp(left_start, right_start, matches.data(), indices.data(),
                        kNumElements);
    }
  }
}

static void RandomCompareMemcmpAligned(benchmark::State& state) {
  RandomCompare<false, true>(state);
}
static void RandomCompareMemcmpUnaligned(benchmark::State& state) {
  RandomCompare<false, false>(state);
}
static void RandomCompareCast(benchmark::State& state) {
  RandomCompare<true, true>(state);
}

// ----------------------------------------------------------------------
// Benchmark declarations

BENCHMARK(HashIntegers);
BENCHMARK(HashSmallStrings);
BENCHMARK(HashMediumStrings);
BENCHMARK(HashLargeStrings);
BENCHMARK(RandomCompareMemcmpAligned);
BENCHMARK(RandomCompareMemcmpUnaligned);
BENCHMARK(RandomCompareCast);

}  // namespace internal
}  // namespace arrow
