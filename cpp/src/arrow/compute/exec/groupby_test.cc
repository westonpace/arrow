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
#include "arrow/compute/exec/groupby.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <memory>

#include "arrow/testing/gtest_util.h"

namespace arrow {

namespace compute {

TEST(GroupByConvenienceFunc, Basic) {
  std::shared_ptr<Array> key1 = ArrayFromJSON(utf8(), R"(["x", "y", "y", "z", "z"])");
  std::shared_ptr<Array> key2 = ArrayFromJSON(int32(), R"([1, 1, 2, 2, 2])");
  std::shared_ptr<Array> values = ArrayFromJSON(int32(), R"([1, 2, 3, 4, 5])");

  // One key, two aggregates, same values array
  std::shared_ptr<Table> expected = TableFromJSON(
      schema({field("sum", int64()), field("count", int64()), field("key_0", utf8())}),
      {
          R"([
        [1, 1, "x"],
        [5, 2, "y"],
        [9, 2, "z"]
    ])"});
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Table> actual,
                       GroupBy({values, values}, {key1}, {},
                               {{"hash_sum", nullptr}, {"hash_count", nullptr}}));
  AssertTablesEqual(*expected, *actual);

  // Two keys, one aggregate
  expected = TableFromJSON(
      schema({field("sum", int64()), field("key_0", utf8()), field("key_1", int32())}),
      {
          R"([
        [1, "x", 1],
        [2, "y", 1],
        [3, "y", 2],
        [9, "z", 2]
      ])"});

  ASSERT_OK_AND_ASSIGN(actual,
                       GroupBy({values}, {key1, key2}, {}, {{"hash_sum", nullptr}}));
  AssertTablesEqual(*expected, *actual);

  // No keys (whole table aggregate)
  expected =
      TableFromJSON(schema({field("sum", int64()), field("count", int64())}), {
                                                                                  R"([
      [15, 5]
    ])"});
  ASSERT_OK_AND_ASSIGN(
      actual, GroupBy({values, values}, {}, {}, {{"sum", nullptr}, {"count", nullptr}}));

  // No aggregates (used to key distinct key values)
  expected =
      TableFromJSON(schema({field("key_0", utf8()), field("key_1", int32())}), {
                                                                                   R"([
      ["x", 1],
      ["y", 1],
      ["y", 2],
      ["z", 2]
    ])"});
  ASSERT_OK_AND_ASSIGN(actual, GroupBy({}, {key1, key2}, {}, {}));
  AssertTablesEqual(*expected, *actual);
}

TEST(GroupByConvenienceFunc, Invalid) {
  std::shared_ptr<Array> key1 = ArrayFromJSON(utf8(), R"(["x", "y", "y", "z", "z"])");
  std::shared_ptr<Array> key2 = ArrayFromJSON(int32(), R"([1, 1, 2, 2, 2])");
  std::shared_ptr<Array> values = ArrayFromJSON(int32(), R"([1, 2, 3, 4, 5])");

  // Mismatching # of aggregates and values
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, ::testing::HasSubstr("arguments and aggregates must be the same size"),
      GroupBy({values}, {}, {}, {{"sum", nullptr}, {"count", nullptr}}));

  // Different length inputs
  std::shared_ptr<Array> short_arr = ArrayFromJSON(int32(), R"([1])");
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, ::testing::HasSubstr("all inputs to GroupBy must have the same length"),
      GroupBy({values}, {short_arr}, {}, {{"count", nullptr}}));
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, ::testing::HasSubstr("all inputs to GroupBy must have the same length"),
      GroupBy({short_arr}, {key1}, {}, {{"count", nullptr}}));

  // Improper function name
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, ::testing::HasSubstr("Use the variant starting with hash_ instead"),
      GroupBy({values}, {key1}, {}, {{"count", nullptr}}));
  EXPECT_RAISES_WITH_MESSAGE_THAT(
      Invalid, ::testing::HasSubstr("Use the variant without hash_ instead"),
      GroupBy({values}, {}, {}, {{"hash_count", nullptr}}));
}

}  // namespace compute
}  // namespace arrow
