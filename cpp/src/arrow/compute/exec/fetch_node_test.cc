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

#include <gtest/gtest.h>

#include "arrow/compute/exec/exec_plan.h"
#include "arrow/compute/exec/options.h"
#include "arrow/compute/exec/test_nodes.h"
#include "arrow/table.h"
#include "arrow/testing/generator.h"
#include "arrow/testing/random.h"

namespace arrow {
namespace compute {

static constexpr int kRowsPerBatch = 16;
static constexpr int kNumBatches = 32;

std::shared_ptr<Table> TestTable() {
  return gen::TestGen({gen::Step()})->Table(kRowsPerBatch, kNumBatches);
}

void CheckFetch(FetchNodeOptions options) {
  constexpr random::SeedType kSeed = 42;
  constexpr int kJitterMod = 4;
  RegisterTestNodes();
  std::shared_ptr<Table> input = TestTable();
  Declaration plan =
      Declaration::Sequence({{"table_source", TableSourceNodeOptions(input)},
                             {"jitter", JitterNodeOptions(kSeed, kJitterMod)},
                             {"fetch", options}});
  PlanExecutionOptions do_sequence;
  do_sequence.sequence_output = true;
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Table> actual,
                       DeclarationToTable(std::move(plan), do_sequence));

  std::shared_ptr<Table> expected = input->Slice(options.offset, options.count);
  AssertTablesEqual(*expected, *actual);
}

TEST(FetchNode, Basic) {
  // CheckFetch({0, 20});
  CheckFetch({20, 20});
}

}  // namespace compute
}  // namespace arrow