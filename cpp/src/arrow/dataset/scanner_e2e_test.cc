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

#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "arrow/array/builder_primitive.h"
#include "arrow/compute/exec/test_util.h"
#include "arrow/csv/writer.h"
#include "arrow/filesystem/localfs.h"
#include "arrow/result.h"
#include "arrow/table.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/random.h"
#include "arrow/util/io_util.h"
#include "arrow/util/logging.h"
#include "arrow/util/macros.h"
#include "arrow/util/string.h"
#include "arrow/util/value_parsing.h"

namespace arrow {
namespace dataset {

struct TestDataConfiguration {
  int num_files;
  int num_partitions;
  int batches_per_file;
  int rows_per_batch;
  random::SeedType seed;

  bool MatchesAllButSeed(const TestDataConfiguration& other) const {
    return other.num_files == num_files && other.num_partitions == num_partitions &&
           other.batches_per_file == batches_per_file &&
           other.rows_per_batch == rows_per_batch;
  }

  bool Matches(const TestDataConfiguration& other) const {
    return MatchesAllButSeed(other) && other.seed == seed;
  }
};

TestDataConfiguration GetDefaultTestConfig() {
  return {/*num_files=*/20,
          /*num_partitions=*/4,
          /*batches_per_file=*/32,
          /*rows_per_batch=*/1024,
          /*seed=*/0};
}

TestDataConfiguration GetLargeTestConfig() {
  return {/*num_files=*/20,
          /*num_partitions=*/4,
          /*batches_per_file=*/32,
          /*rows_per_batch=*/1024 * 1024,
          /*seed=*/0};
}

TestDataConfiguration GetTestDataConfigurationForMode(const std::string& test_mode) {
  if (test_mode == "LARGE") {
    return GetLargeTestConfig();
  } else if (test_mode == "DEFAULT") {
    return GetDefaultTestConfig();
  } else {
    DCHECK(false) << "expected SCANNER_TEST_MODE to be LARGE or DEFAULT but was '"
                  << test_mode << "'";
    return GetDefaultTestConfig();
  }
}

std::string GetDesiredTestMode() {
  Result<std::string> maybe_test_mode = internal::GetEnvVar("SCANNER_TEST_MODE");
  if (!maybe_test_mode.ok()) {
    return "DEFAULT";
  }
  return *maybe_test_mode;
}

fs::LocalFileSystem kLocalFileSystem;

std::optional<TestDataConfiguration> LoadCurrentTestDataConfiguration(
    const std::string& test_data_dir) {
  fs::FileSelector selector;
  selector.base_dir = test_data_dir;
  selector.allow_not_found = true;
  Result<fs::FileInfoVector> maybe_file_infos = kLocalFileSystem.GetFileInfo(selector);
  if (maybe_file_infos.ok()) {
    return std::nullopt;
  }

  for (const auto& file_info : *maybe_file_infos) {
    if (internal::EndsWith(file_info.base_name(), ".params")) {
      std::vector<std::string_view> params =
          internal::SplitString(file_info.base_name(), '-');
      if (params.size() != 5) {
        return std::nullopt;
      }
      TestDataConfiguration data_config;
      if (!internal::ParseValue<Int32Type>(params[0].data(), params[0].size(),
                                           &data_config.num_partitions)) {
        return std::nullopt;
      }
      if (!internal::ParseValue<Int32Type>(params[1].data(), params[1].size(),
                                           &data_config.num_files)) {
        return std::nullopt;
      }
      if (!internal::ParseValue<Int32Type>(params[2].data(), params[2].size(),
                                           &data_config.batches_per_file)) {
        return std::nullopt;
      }
      if (!internal::ParseValue<Int32Type>(params[3].data(), params[3].size(),
                                           &data_config.rows_per_batch)) {
        return std::nullopt;
      }
      if (!internal::ParseValue<Int32Type>(params[4].data(), params[4].size(),
                                           &data_config.seed)) {
        return std::nullopt;
      }
      return data_config;
    }
  }
  return std::nullopt;
}

class FormatUtilities {
 public:
  virtual ~FormatUtilities() = default;
  virtual Status WriteTable(const Table& table, const std::string& destination) const = 0;
  virtual bool supports_statistics() const = 0;
  virtual std::string_view name() const = 0;
};

class CsvFormatUtilities : public FormatUtilities {
 public:
  ~CsvFormatUtilities() override = default;
  Status WriteTable(const Table& table, const std::string& destination) const override {
    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<io::OutputStream> out,
                          kLocalFileSystem.OpenOutputStream(destination));
    return csv::WriteCSV(table, /*options=*/{}, out.get());
  }

  bool supports_statistics() const override { return false; }
  std::string_view name() const override { return "csv"; }
};

const std::vector<std::shared_ptr<FormatUtilities>> kFormats = {
    std::make_shared<CsvFormatUtilities>()};

const std::shared_ptr<Schema> kTestSchema = schema({field("id", int64())});

Result<std::shared_ptr<Array>> CreateIdArray(TestDataConfiguration config,
                                             int batch_num) {
  Int64Builder builder;
  ARROW_RETURN_NOT_OK(builder.Reserve(config.rows_per_batch));
  for (int64_t row_num = batch_num * config.rows_per_batch;
       row_num < (batch_num + 1) * config.rows_per_batch; row_num++) {
    ARROW_RETURN_NOT_OK(builder.Append(row_num));
  }
  std::shared_ptr<Array> ids;
  ARROW_RETURN_NOT_OK(builder.Finish(&ids));
  return ids;
}

Result<std::shared_ptr<RecordBatch>> CreateTestBatch(TestDataConfiguration config,
                                                     int batch_num) {
  std::vector<std::shared_ptr<Array>> arrays;
  ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Array> ids, CreateIdArray(config, batch_num));
  arrays.push_back(std::move(ids));
  return RecordBatch::Make(kTestSchema, config.rows_per_batch, std::move(arrays));
}

Result<std::shared_ptr<Table>> CreateTestTable(TestDataConfiguration config,
                                               int* batch_num,
                                               random::RandomArrayGenerator* gen) {
  std::vector<std::shared_ptr<RecordBatch>> batches;
  for (int i = 0; i < config.batches_per_file; i++) {
    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<RecordBatch> batch,
                          CreateTestBatch(config, *batch_num));
    *batch_num = *batch_num + 1;
    batches.push_back(std::move(batch));
  }
  return Table::FromRecordBatches(batches);
}

Status CreateTestData(const std::string& test_data_dir, TestDataConfiguration config) {
  random::RandomArrayGenerator gen(config.seed);
  ARROW_RETURN_NOT_OK(kLocalFileSystem.CreateDir(test_data_dir));
  internal::PlatformFilename data_dir(test_data_dir);
  int files_per_partition =
      static_cast<int>(bit_util::CeilDiv(config.num_files, config.num_partitions));
  int files_created = 0;
  int batches_created = 0;
  for (int part = 0; part < config.num_partitions; part++) {
    for (const auto& format : kFormats) {
      ARROW_ASSIGN_OR_RAISE(auto format_dir,
                            data_dir.Join(std::string(format->name()) +
                                          "/part=" + internal::ToChars(part)));
      ARROW_RETURN_NOT_OK(kLocalFileSystem.CreateDir(format_dir.ToString()));
    }
    for (int filenum = 0; filenum < files_per_partition; filenum++) {
      ARROW_ASSIGN_OR_RAISE(std::shared_ptr<Table> table,
                            CreateTestTable(config, &batches_created, &gen));
      for (const auto& format : kFormats) {
        ARROW_ASSIGN_OR_RAISE(auto file_path,
                              data_dir.Join(std::string(format->name()) +
                                            "/part=" + internal::ToChars(part) + "/" +
                                            internal::ToChars(filenum) + "." +
                                            std::string(format->name())));
        ARROW_RETURN_NOT_OK(format->WriteTable(*table, file_path.ToString()));
      }
      if (++files_created == config.num_files) {
        return Status::OK();
      }
    }
  }
  return Status::OK();
}

Result<random::SeedType> EnsureTestData(std::optional<random::SeedType> desired_seed,
                                        const std::string& desired_mode,
                                        const std::string& test_data_dir) {
  std::optional<TestDataConfiguration> existing =
      LoadCurrentTestDataConfiguration(test_data_dir);
  TestDataConfiguration desired_config = GetTestDataConfigurationForMode(desired_mode);
  if (desired_seed) {
    desired_config.seed = *desired_seed;
  } else {
    desired_config.seed = random::SeedFromTime();
  }
  // First, if the test data already exists with the same config & seed (don't require
  // same seed if user didn't ask for a specific seed) then use that.
  if (existing) {
    if (desired_seed) {
      if (existing->Matches(desired_config)) {
        return desired_config.seed;
      }
    } else if (existing->MatchesAllButSeed(desired_config)) {
      return desired_config.seed;
    }
  }

  // If there is no test data or it didn't match then delete whatever exists and create
  ARROW_RETURN_NOT_OK(kLocalFileSystem.DeleteDir(test_data_dir));
  ARROW_RETURN_NOT_OK(CreateTestData(test_data_dir, desired_config));
  return desired_config.seed;
}

std::optional<std::string> GetUserSpecifiedTestDataDir() {
  Result<std::string> maybe_test_dir = internal::GetEnvVar("SCANNER_TEST_DIR");
  if (maybe_test_dir.ok()) {
    return *maybe_test_dir;
  }
  return std::nullopt;
}

std::optional<random::SeedType> GetUserSpecifiedSeed() {
  Result<std::string> maybe_seed = internal::GetEnvVar("SCANNER_SEED");
  if (!maybe_seed.ok()) {
    return std::nullopt;
  }
  const std::string& seed_str = *maybe_seed;
  random::SeedType seed;
  if (!internal::ParseValue<Int32Type>(seed_str.data(), seed_str.size(), &seed)) {
    ADD_FAILURE() << "could not parse " << seed_str << " as a seed";
    return std::nullopt;
  }
  return seed;
}

class ScannerTestEnvironment : public ::testing::Environment {
 public:
  ~ScannerTestEnvironment() override {}

  void SetUp() override {
    if (!GetUserSpecifiedTestDataDir()) {
      ASSERT_OK_AND_ASSIGN(tempdir, internal::TemporaryDir::Make("scanner_e2e"));
      ASSERT_OK(internal::SetEnvVar("SCANNER_TEST_DIR", tempdir->path().ToString()));
    }
  }

  void TearDown() override {}

 private:
  std::unique_ptr<internal::TemporaryDir> tempdir;
};

::testing::Environment* scanner_test_env =
    ::testing::AddGlobalTestEnvironment(new ScannerTestEnvironment());

class ScannerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::optional<random::SeedType> desired_seed = GetUserSpecifiedSeed();
    std::string desired_test_mode = GetDesiredTestMode();
    std::optional<std::string> desired_data_dir = GetUserSpecifiedTestDataDir();
    if (!desired_data_dir) {
      FAIL() << "expected data directory to be initialized by test environment";
      return;
    }
    datadir = *desired_data_dir;
    ASSERT_OK_AND_ASSIGN(datadir_seed,
                         EnsureTestData(desired_seed, desired_test_mode, datadir));
  }

  TestDataConfiguration test_data_configuration;
  random::SeedType datadir_seed;
  std::string datadir;
};

enum class RowSelectionStrategy { kNoRows = 0, kSomeRows = 1, kOneRow = 2, kAllRows = 3 };

enum class FilterStrategy {
  kNoFilter,
  kFilterOnPartition,
  kFilterOnStatistics,
  kFilterOnPartitionAndStatistics
};

struct TestConfiguration {
  bool simulate_io_errors;
  RowSelectionStrategy row_selection_strategy;
  FilterStrategy filter_strategy;
  FormatUtilities* format;
};

bool ShouldSimulateIoErrors(compute::Random64Bit* gen) {
  // 5% chance we simulate I/O errors
  return gen->from_range<int>(0, 20) == 0;
}

FormatUtilities* PickFormat(compute::Random64Bit* gen) {
  std::size_t choice = gen->from_range<std::size_t>(0, kFormats.size() - 1);
  return kFormats[choice].get();
}

RowSelectionStrategy PickRowSelectionStrategy(compute::Random64Bit* gen) {
  int choice = gen->from_range<int>(0, 3);
  return static_cast<RowSelectionStrategy>(choice);
}

FilterStrategy PickFilterStrategy(compute::Random64Bit* gen, FormatUtilities* format) {
  if (format->supports_statistics()) {
    int choice = gen->from_range<int>(0, 3);
    return static_cast<FilterStrategy>(choice);
  }
  int has_filter = gen->from_range<int>(0, 1);
  if (has_filter == 0) {
    return FilterStrategy::kNoFilter;
  } else {
    return FilterStrategy::kFilterOnPartition;
  }
}

TestConfiguration PickRandomConfiguration(random::SeedType seed) {
  compute::Random64Bit gen(seed);
  TestConfiguration test_configuration;
  test_configuration.simulate_io_errors = ShouldSimulateIoErrors(&gen);
  test_configuration.format = PickFormat(&gen);
  test_configuration.row_selection_strategy = PickRowSelectionStrategy(&gen);
  test_configuration.filter_strategy =
      PickFilterStrategy(&gen, test_configuration.format);
  return test_configuration;
}

TEST_F(ScannerTest, Stress) {
  constexpr int kNumTests = 100;
  ARROW_SCOPED_TRACE("data seed=", datadir_seed, " datadir=", datadir);
  random::RandomArrayGenerator seed_gen(random::SeedFromTime());
  for (int i = 0; i < kNumTests; i++) {
    random::SeedType test_seed = seed_gen.seed();
    TestConfiguration test_config = PickRandomConfiguration(test_seed);
    std::cout << "ioerr=" << test_config.simulate_io_errors
              << " fmt=" << test_config.format->name()
              << " rows=" << static_cast<int>(test_config.row_selection_strategy)
              << " filt=" << static_cast<int>(test_config.filter_strategy) << std::endl;
  }
}

}  // namespace dataset
}  // namespace arrow
