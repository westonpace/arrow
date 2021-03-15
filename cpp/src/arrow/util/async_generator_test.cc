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

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "arrow/testing/gtest_util.h"
#include "arrow/type_fwd.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/test_common.h"
#include "arrow/util/vector.h"

namespace arrow {

template <typename T>
AsyncGenerator<T> AsyncVectorIt(std::vector<T> v) {
  return MakeVectorGenerator(std::move(v));
}

template <typename T>
AsyncGenerator<T> Slowdown(AsyncGenerator<T> source, double seconds) {
  return MakeMappedGenerator<T, T>(
      std::move(source), [seconds](const T& res) -> Future<T> {
        return SleepAsync(seconds).Then(
            [res](const Result<detail::Empty>& empty) { return res; });
      });
}

template <typename T>
AsyncGenerator<T> SlowdownABit(AsyncGenerator<T> source) {
  return Slowdown(std::move(source), 1e-3);
}

template <typename T>
class TrackingGenerator {
 public:
  explicit TrackingGenerator(AsyncGenerator<T> source)
      : state_(std::make_shared<State>(std::move(source))) {}

  Future<T> operator()() {
    state_->num_read++;
    return state_->source();
  }

  int num_read() { return state_->num_read; }

 private:
  struct State {
    explicit State(AsyncGenerator<T> source) : source(std::move(source)), num_read(0) {}

    AsyncGenerator<T> source;
    int num_read;
  };

  std::shared_ptr<State> state_;
};

constexpr auto kYieldDuration = std::chrono::microseconds(50);

// Yields items with a small pause between each one from a background thread
std::function<Future<TestInt>()> BackgroundAsyncVectorIt(std::vector<TestInt> v,
                                                         bool sleep = true) {
  auto pool = internal::GetCpuThreadPool();
  auto iterator = VectorIt(v);
  auto slow_iterator = MakeTransformedIterator<TestInt, TestInt>(
      std::move(iterator), [sleep](TestInt item) -> Result<TransformFlow<TestInt>> {
        if (sleep) {
          std::this_thread::sleep_for(kYieldDuration);
        }
        return TransformYield(item);
      });

  EXPECT_OK_AND_ASSIGN(auto background,
                       MakeBackgroundGenerator<TestInt>(std::move(slow_iterator),
                                                        internal::GetCpuThreadPool()));
  return MakeTransferredGenerator(background, pool);
}

template <typename T>
void AssertAsyncGeneratorMatch(std::vector<T> expected, AsyncGenerator<T> actual) {
  auto vec_future = CollectAsyncGenerator(std::move(actual));
  EXPECT_OK_AND_ASSIGN(auto vec, vec_future.result());
  EXPECT_EQ(expected, vec);
}

template <typename T>
void AssertGeneratorExhausted(AsyncGenerator<T>& gen) {
  ASSERT_FINISHES_OK_AND_ASSIGN(auto next, gen());
  ASSERT_TRUE(IsIterationEnd(next));
}

// --------------------------------------------------------------------
// Asynchronous iterator tests

template <typename T>
class ReentrantCheckerGuard;

template <typename T>
ReentrantCheckerGuard<T> ExpectNotAccessedReentrantly(AsyncGenerator<T>* generator);

template <typename T>
class ReentrantChecker {
 public:
  Future<T> operator()() {
    if (state_->generated_unfinished_future.load()) {
      state_->valid.store(false);
    }
    state_->generated_unfinished_future.store(true);
    auto result = state_->source();
    return result.Then(Callback{state_});
  }

  bool valid() { return state_->valid.load(); }

 private:
  explicit ReentrantChecker(AsyncGenerator<T> source)
      : state_(std::make_shared<State>(std::move(source))) {}

  friend ReentrantCheckerGuard<T> ExpectNotAccessedReentrantly<T>(
      AsyncGenerator<T>* generator);

  struct State {
    explicit State(AsyncGenerator<T> source_)
        : source(std::move(source_)), generated_unfinished_future(false), valid(true) {}

    AsyncGenerator<T> source;
    std::atomic<bool> generated_unfinished_future;
    std::atomic<bool> valid;
  };
  struct Callback {
    Future<T> operator()(const Result<T>& result) {
      state_->generated_unfinished_future.store(false);
      return result;
    }
    std::shared_ptr<State> state_;
  };

  std::shared_ptr<State> state_;
};

template <typename T>
class ReentrantCheckerGuard {
 public:
  explicit ReentrantCheckerGuard(ReentrantChecker<T> checker) : checker_(checker) {}

  ARROW_DISALLOW_COPY_AND_ASSIGN(ReentrantCheckerGuard);
  ReentrantCheckerGuard(ReentrantCheckerGuard&& other) : checker_(other.checker_) {
    if (other.owner_) {
      other.owner_ = false;
      owner_ = true;
    } else {
      owner_ = false;
    }
  }
  ReentrantCheckerGuard& operator=(ReentrantCheckerGuard&& other) {
    checker_ = other.checker_;
    if (other.owner_) {
      other.owner_ = false;
      owner_ = true;
    } else {
      owner_ = false;
    }
    return *this;
  }

  ~ReentrantCheckerGuard() {
    if (owner_ && !checker_.valid()) {
      ADD_FAILURE() << "A generator was accessed reentrantly when the test asserted it "
                       "should not be.";
    }
  }

 private:
  ReentrantChecker<T> checker_;
  bool owner_ = true;
};

template <typename T>
ReentrantCheckerGuard<T> ExpectNotAccessedReentrantly(AsyncGenerator<T>* generator) {
  auto reentrant_checker = ReentrantChecker<T>(*generator);
  *generator = reentrant_checker;
  return ReentrantCheckerGuard<T>(reentrant_checker);
}

TEST(TestAsyncUtil, Visit) {
  auto generator = AsyncVectorIt<TestInt>({1, 2, 3});
  unsigned int sum = 0;
  auto sum_future = VisitAsyncGenerator<TestInt>(generator, [&sum](TestInt item) {
    sum += item.value;
    return Status::OK();
  });
  ASSERT_TRUE(sum_future.is_finished());
  ASSERT_EQ(6, sum);
}

TEST(TestAsyncUtil, Collect) {
  std::vector<TestInt> expected = {1, 2, 3};
  auto generator = AsyncVectorIt(expected);
  auto collected = CollectAsyncGenerator(generator);
  ASSERT_FINISHES_OK_AND_ASSIGN(auto collected_val, collected);
  ASSERT_EQ(expected, collected_val);
}

TEST(TestAsyncUtil, Map) {
  std::vector<TestInt> input = {1, 2, 3};
  auto generator = AsyncVectorIt(input);
  std::function<TestStr(const TestInt&)> mapper = [](const TestInt& in) {
    return std::to_string(in.value);
  };
  auto mapped = MakeMappedGenerator(std::move(generator), mapper);
  std::vector<TestStr> expected{"1", "2", "3"};
  AssertAsyncGeneratorMatch(expected, mapped);
}

TEST(TestAsyncUtil, MapAsync) {
  std::vector<TestInt> input = {1, 2, 3};
  auto generator = AsyncVectorIt(input);
  std::function<Future<TestStr>(const TestInt&)> mapper = [](const TestInt& in) {
    return SleepAsync(1e-3).Then([in](const Result<detail::Empty>& empty) {
      return TestStr(std::to_string(in.value));
    });
  };
  auto mapped = MakeMappedGenerator(std::move(generator), mapper);
  std::vector<TestStr> expected{"1", "2", "3"};
  AssertAsyncGeneratorMatch(expected, mapped);
}

TEST(TestAsyncUtil, MapReentrant) {
  std::vector<TestInt> input = {1, 2};
  auto source = AsyncVectorIt(input);
  TrackingGenerator<TestInt> tracker(std::move(source));
  source = MakeTransferredGenerator(AsyncGenerator<TestInt>(tracker),
                                    internal::GetCpuThreadPool());

  std::atomic<int> map_tasks_running(0);
  // Mapper blocks until can_proceed is marked finished, should start multiple map tasks
  Future<> can_proceed = Future<>::Make();
  std::function<Future<TestStr>(const TestInt&)> mapper = [&](const TestInt& in) {
    map_tasks_running.fetch_add(1);
    return can_proceed.Then([in](...) { return TestStr(std::to_string(in.value)); });
  };
  auto mapped = MakeMappedGenerator(std::move(source), mapper);

  EXPECT_EQ(0, tracker.num_read());

  auto one = mapped();
  auto two = mapped();

  BusyWait(10, [&] { return map_tasks_running.load() == 2; });
  EXPECT_EQ(2, map_tasks_running.load());
  EXPECT_EQ(2, tracker.num_read());

  auto end_one = mapped();
  auto end_two = mapped();

  can_proceed.MarkFinished();
  ASSERT_FINISHES_OK_AND_ASSIGN(auto oneval, one);
  EXPECT_EQ("1", oneval.value);
  ASSERT_FINISHES_OK_AND_ASSIGN(auto twoval, two);
  EXPECT_EQ("2", twoval.value);
  ASSERT_FINISHES_OK_AND_ASSIGN(auto end, end_one);
  ASSERT_EQ(IterationTraits<TestStr>::End(), end);
  ASSERT_FINISHES_OK_AND_ASSIGN(end, end_two);
  ASSERT_EQ(IterationTraits<TestStr>::End(), end);
}

TEST(TestAsyncUtil, MapParallelStress) {
  constexpr int NTASKS = 10;
  constexpr int NITEMS = 10;
  for (int i = 0; i < NTASKS; i++) {
    auto gen = MakeVectorGenerator(RangeVector(NITEMS));
    gen = SlowdownABit(std::move(gen));
    auto guard = ExpectNotAccessedReentrantly(&gen);
    std::function<TestStr(const TestInt&)> mapper = [](const TestInt& in) {
      SleepABit();
      return std::to_string(in.value);
    };
    auto mapped = MakeMappedGenerator(std::move(gen), mapper);
    mapped = MakeReadaheadGenerator(mapped, 8);
    ASSERT_FINISHES_OK_AND_ASSIGN(auto collected, CollectAsyncGenerator(mapped));
    ASSERT_EQ(NITEMS, collected.size());
  }
}

TEST(TestAsyncUtil, MaybeMapFail) {
  std::vector<TestInt> input = {1, 2, 3};
  auto generator = AsyncVectorIt(input);
  std::function<Result<TestStr>(const TestInt&)> mapper =
      [](const TestInt& in) -> Result<TestStr> {
    if (in.value == 2) {
      return Status::Invalid("XYZ");
    }
    return TestStr(std::to_string(in.value));
  };
  auto mapped = MakeMappedGenerator(std::move(generator), mapper);
  ASSERT_FINISHES_ERR(Invalid, CollectAsyncGenerator(mapped));
}

TEST(TestAsyncUtil, Concatenated) {
  std::vector<TestInt> inputOne{1, 2, 3};
  std::vector<TestInt> inputTwo{4, 5, 6};
  std::vector<TestInt> expected{1, 2, 3, 4, 5, 6};
  auto gen = AsyncVectorIt<AsyncGenerator<TestInt>>(
      {AsyncVectorIt<TestInt>(inputOne), AsyncVectorIt<TestInt>(inputTwo)});
  auto concat = MakeConcatenatedGenerator(gen);
  AssertAsyncGeneratorMatch(expected, concat);
}

class GeneratorTestFixture : public ::testing::TestWithParam<bool> {
 protected:
  AsyncGenerator<TestInt> MakeSource(const std::vector<TestInt>& items) {
    std::vector<TestInt> wrapped(items.begin(), items.end());
    auto gen = AsyncVectorIt(std::move(wrapped));
    bool slow = GetParam();
    if (slow) {
      return SlowdownABit(std::move(gen));
    }
    return gen;
  }

  int GetNumItersForStress() {
    bool slow = GetParam();
    // Run fewer trials for the slow case since they take longer
    if (slow) {
      return 10;
    } else {
      return 100;
    }
  }
};

TEST_P(GeneratorTestFixture, Merged) {
  auto gen = AsyncVectorIt<AsyncGenerator<TestInt>>(
      {MakeSource({1, 2, 3}), MakeSource({4, 5, 6})});

  auto concat_gen = MakeMergedGenerator(gen, 10);
  ASSERT_FINISHES_OK_AND_ASSIGN(auto concat, CollectAsyncGenerator(concat_gen));
  auto concat_ints =
      internal::MapVector([](const TestInt& val) { return val.value; }, concat);
  std::set<int> concat_set(concat_ints.begin(), concat_ints.end());

  std::set<int> expected{1, 2, 4, 3, 5, 6};
  ASSERT_EQ(expected, concat_set);
}

TEST_P(GeneratorTestFixture, MergedLimitedSubscriptions) {
  auto gen = AsyncVectorIt<AsyncGenerator<TestInt>>(
      {MakeSource({1, 2}), MakeSource({3, 4}), MakeSource({5, 6, 7, 8}),
       MakeSource({9, 10, 11, 12})});
  TrackingGenerator<AsyncGenerator<TestInt>> tracker(std::move(gen));
  auto merged = MakeMergedGenerator(AsyncGenerator<AsyncGenerator<TestInt>>(tracker), 2);

  SleepABit();
  // Lazy pull, should not start until first pull
  ASSERT_EQ(0, tracker.num_read());

  ASSERT_FINISHES_OK_AND_ASSIGN(auto next, merged());
  ASSERT_TRUE(next.value == 1 || next.value == 3);

  // First 2 values have to come from one of the first 2 sources
  ASSERT_EQ(2, tracker.num_read());
  ASSERT_FINISHES_OK_AND_ASSIGN(next, merged());
  ASSERT_LT(next.value, 5);
  ASSERT_GT(next.value, 0);

  // By the time five values have been read we should have exhausted at
  // least one source
  for (int i = 0; i < 3; i++) {
    ASSERT_FINISHES_OK_AND_ASSIGN(next, merged());
    // 9 is possible if we read 1,2,3,4 and then grab 9 while 5 is running slow
    ASSERT_LT(next.value, 10);
    ASSERT_GT(next.value, 0);
  }
  ASSERT_GT(tracker.num_read(), 2);
  ASSERT_LT(tracker.num_read(), 5);

  // Read remaining values
  for (int i = 0; i < 7; i++) {
    ASSERT_FINISHES_OK_AND_ASSIGN(next, merged());
    ASSERT_LT(next.value, 13);
    ASSERT_GT(next.value, 0);
  }

  AssertGeneratorExhausted(merged);
}

TEST_P(GeneratorTestFixture, MergedStress) {
  constexpr int NGENERATORS = 10;
  constexpr int NITEMS = 10;
  for (int i = 0; i < GetNumItersForStress(); i++) {
    std::vector<AsyncGenerator<TestInt>> sources;
    std::vector<ReentrantCheckerGuard<TestInt>> guards;
    for (int j = 0; j < NGENERATORS; j++) {
      auto source = MakeSource(RangeVector(NITEMS));
      guards.push_back(ExpectNotAccessedReentrantly(&source));
      sources.push_back(source);
    }
    AsyncGenerator<AsyncGenerator<TestInt>> source_gen = AsyncVectorIt(sources);

    auto merged = MakeMergedGenerator(source_gen, 4);
    ASSERT_FINISHES_OK_AND_ASSIGN(auto items, CollectAsyncGenerator(merged));
    ASSERT_EQ(NITEMS * NGENERATORS, items.size());
  }
}

TEST_P(GeneratorTestFixture, MergedParallelStress) {
  constexpr int NGENERATORS = 10;
  constexpr int NITEMS = 10;
  for (int i = 0; i < GetNumItersForStress(); i++) {
    std::vector<AsyncGenerator<TestInt>> sources;
    for (int j = 0; j < NGENERATORS; j++) {
      sources.push_back(MakeSource(RangeVector(NITEMS)));
    }
    auto merged = MakeMergedGenerator(AsyncVectorIt(sources), 4);
    merged = MakeReadaheadGenerator(merged, 4);
    ASSERT_FINISHES_OK_AND_ASSIGN(auto items, CollectAsyncGenerator(merged));
    ASSERT_EQ(NITEMS * NGENERATORS, items.size());
  }
}

INSTANTIATE_TEST_SUITE_P(GeneratorTests, GeneratorTestFixture,
                         ::testing::Values(false, true));

TEST(TestAsyncUtil, FromVector) {
  AsyncGenerator<TestInt> gen;
  {
    std::vector<TestInt> input = {1, 2, 3};
    gen = MakeVectorGenerator(std::move(input));
  }
  std::vector<TestInt> expected = {1, 2, 3};
  AssertAsyncGeneratorMatch(expected, gen);
}

TEST(TestAsyncUtil, SynchronousFinish) {
  AsyncGenerator<TestInt> generator = []() {
    return Future<TestInt>::MakeFinished(IterationTraits<TestInt>::End());
  };
  Transformer<TestInt, TestStr> skip_all = [](TestInt value) { return TransformSkip(); };
  auto transformed = MakeAsyncGenerator(generator, skip_all);
  auto future = CollectAsyncGenerator(transformed);
  ASSERT_FINISHES_OK_AND_ASSIGN(auto actual, future);
  ASSERT_EQ(std::vector<TestStr>(), actual);
}

TEST(TestAsyncUtil, GeneratorIterator) {
  auto generator = BackgroundAsyncVectorIt({1, 2, 3});
  ASSERT_OK_AND_ASSIGN(auto iterator, MakeGeneratorIterator(std::move(generator)));
  ASSERT_OK_AND_EQ(TestInt(1), iterator.Next());
  ASSERT_OK_AND_EQ(TestInt(2), iterator.Next());
  ASSERT_OK_AND_EQ(TestInt(3), iterator.Next());
  AssertIteratorExhausted(iterator);
  AssertIteratorExhausted(iterator);
}

TEST(TestAsyncUtil, MakeTransferredGenerator) {
  std::mutex mutex;
  std::condition_variable cv;
  std::atomic<bool> finished(false);

  ASSERT_OK_AND_ASSIGN(auto thread_pool, internal::ThreadPool::Make(1));

  // Needs to be a slow source to ensure we don't call Then on a completed
  AsyncGenerator<TestInt> slow_generator = [&]() {
    return thread_pool
        ->Submit([&] {
          std::unique_lock<std::mutex> lock(mutex);
          cv.wait_for(lock, std::chrono::duration<double>(30),
                      [&] { return finished.load(); });
          return IterationTraits<TestInt>::End();
        })
        .ValueOrDie();
  };

  auto transferred =
      MakeTransferredGenerator<TestInt>(std::move(slow_generator), thread_pool.get());

  auto current_thread_id = std::this_thread::get_id();
  auto fut = transferred().Then([&current_thread_id](const Result<TestInt>& result) {
    ASSERT_NE(current_thread_id, std::this_thread::get_id());
  });

  {
    std::lock_guard<std::mutex> lg(mutex);
    finished.store(true);
  }
  cv.notify_one();
  ASSERT_FINISHES_OK(fut);
}

// This test is too slow for valgrind
#if !(defined(ARROW_VALGRIND) || defined(ADDRESS_SANITIZER))

TEST(TestAsyncUtil, StackOverflow) {
  int counter = 0;
  AsyncGenerator<TestInt> generator = [&counter]() {
    if (counter < 10000) {
      return Future<TestInt>::MakeFinished(counter++);
    } else {
      return Future<TestInt>::MakeFinished(IterationTraits<TestInt>::End());
    }
  };
  Transformer<TestInt, TestStr> discard =
      [](TestInt next) -> Result<TransformFlow<TestStr>> { return TransformSkip(); };
  auto transformed = MakeAsyncGenerator(generator, discard);
  auto collected_future = CollectAsyncGenerator(transformed);
  ASSERT_FINISHES_OK_AND_ASSIGN(auto collected, collected_future);
  ASSERT_EQ(0, collected.size());
}

#endif

TEST(TestAsyncUtil, Background) {
  std::vector<TestInt> expected = {1, 2, 3};
  auto background = BackgroundAsyncVectorIt(expected);
  auto future = CollectAsyncGenerator(background);
  ASSERT_FINISHES_OK_AND_ASSIGN(auto collected, future);
  ASSERT_EQ(expected, collected);
}

struct SlowEmptyIterator {
  Result<TestInt> Next() {
    if (called_) {
      return Status::Invalid("Should not have been called twice");
    }
    SleepFor(0.1);
    return IterationTraits<TestInt>::End();
  }

 private:
  bool called_ = false;
};

TEST(TestAsyncUtil, BackgroundRepeatEnd) {
  // Ensure that the background generator properly fulfills the asyncgenerator contract
  // and can be called after it ends.
  ASSERT_OK_AND_ASSIGN(auto io_pool, internal::ThreadPool::Make(1));

  auto iterator = Iterator<TestInt>(SlowEmptyIterator());
  ASSERT_OK_AND_ASSIGN(auto background_gen,
                       MakeBackgroundGenerator(std::move(iterator), io_pool.get()));

  background_gen =
      MakeTransferredGenerator(std::move(background_gen), internal::GetCpuThreadPool());

  auto one = background_gen();
  auto two = background_gen();

  ASSERT_FINISHES_OK_AND_ASSIGN(auto one_fin, one);
  ASSERT_TRUE(IsIterationEnd(one_fin));

  ASSERT_FINISHES_OK_AND_ASSIGN(auto two_fin, two);
  ASSERT_TRUE(IsIterationEnd(two_fin));
}

TEST(TestAsyncUtil, CompleteBackgroundStressTest) {
  auto expected = RangeVector(20);
  std::vector<Future<std::vector<TestInt>>> futures;
  for (unsigned int i = 0; i < 20; i++) {
    auto background = BackgroundAsyncVectorIt(expected);
    futures.push_back(CollectAsyncGenerator(background));
  }
  auto combined = All(futures);
  ASSERT_FINISHES_OK_AND_ASSIGN(auto completed_vectors, combined);
  for (std::size_t i = 0; i < completed_vectors.size(); i++) {
    ASSERT_OK_AND_ASSIGN(auto vector, completed_vectors[i]);
    ASSERT_EQ(vector, expected);
  }
}

TEST(TestAsyncUtil, SerialReadaheadSlowProducer) {
  AsyncGenerator<TestInt> gen = BackgroundAsyncVectorIt({1, 2, 3, 4, 5});
  auto guard = ExpectNotAccessedReentrantly(&gen);
  SerialReadaheadGenerator<TestInt> serial_readahead(gen, 2);
  AssertAsyncGeneratorMatch({1, 2, 3, 4, 5},
                            static_cast<AsyncGenerator<TestInt>>(serial_readahead));
}

TEST(TestAsyncUtil, SerialReadaheadSlowConsumer) {
  int num_delivered = 0;
  auto source = [&num_delivered]() {
    if (num_delivered < 5) {
      return Future<TestInt>::MakeFinished(num_delivered++);
    } else {
      return Future<TestInt>::MakeFinished(IterationTraits<TestInt>::End());
    }
  };
  SerialReadaheadGenerator<TestInt> serial_readahead(std::move(source), 3);
  SleepABit();
  ASSERT_EQ(0, num_delivered);
  ASSERT_FINISHES_OK_AND_ASSIGN(auto next, serial_readahead());
  ASSERT_EQ(0, next.value);
  ASSERT_EQ(3, num_delivered);
  AssertAsyncGeneratorMatch({1, 2, 3, 4},
                            static_cast<AsyncGenerator<TestInt>>(serial_readahead));
}

TEST(TestAsyncUtil, SerialReadaheadStress) {
  constexpr int NTASKS = 20;
  constexpr int NITEMS = 50;
  for (int i = 0; i < NTASKS; i++) {
    AsyncGenerator<TestInt> gen = BackgroundAsyncVectorIt(RangeVector(NITEMS));
    auto guard = ExpectNotAccessedReentrantly(&gen);
    SerialReadaheadGenerator<TestInt> serial_readahead(gen, 2);
    auto visit_fut =
        VisitAsyncGenerator<TestInt>(serial_readahead, [](TestInt test_int) -> Status {
          // Normally sleeping in a visit function would be a faux-pas but we want to slow
          // the reader down to match the producer to maximize the stress
          std::this_thread::sleep_for(kYieldDuration);
          return Status::OK();
        });
    ASSERT_FINISHES_OK(visit_fut);
  }
}

TEST(TestAsyncUtil, SerialReadaheadStressFast) {
  constexpr int NTASKS = 20;
  constexpr int NITEMS = 50;
  for (int i = 0; i < NTASKS; i++) {
    AsyncGenerator<TestInt> gen = BackgroundAsyncVectorIt(RangeVector(NITEMS), false);
    auto guard = ExpectNotAccessedReentrantly(&gen);
    SerialReadaheadGenerator<TestInt> serial_readahead(gen, 2);
    auto visit_fut = VisitAsyncGenerator<TestInt>(
        serial_readahead, [](TestInt test_int) -> Status { return Status::OK(); });
    ASSERT_FINISHES_OK(visit_fut);
  }
}

TEST(TestAsyncUtil, SerialReadaheadStressFailing) {
  constexpr int NTASKS = 20;
  constexpr int NITEMS = 50;
  constexpr int EXPECTED_SUM = 45;
  for (int i = 0; i < NTASKS; i++) {
    AsyncGenerator<TestInt> it = BackgroundAsyncVectorIt(RangeVector(NITEMS));
    AsyncGenerator<TestInt> fails_at_ten = [&it]() {
      auto next = it();
      return next.Then([](const Result<TestInt>& item) -> Result<TestInt> {
        if (item->value >= 10) {
          return Status::Invalid("XYZ");
        } else {
          return item;
        }
      });
    };
    SerialReadaheadGenerator<TestInt> serial_readahead(fails_at_ten, 2);
    unsigned int sum = 0;
    auto visit_fut = VisitAsyncGenerator<TestInt>(
        serial_readahead, [&sum](TestInt test_int) -> Status {
          sum += test_int.value;
          // Normally sleeping in a visit function would be a faux-pas but we want to slow
          // the reader down to match the producer to maximize the stress
          std::this_thread::sleep_for(kYieldDuration);
          return Status::OK();
        });
    ASSERT_FINISHES_ERR(Invalid, visit_fut);
    ASSERT_EQ(EXPECTED_SUM, sum);
  }
}

TEST(TestAsyncUtil, Readahead) {
  int num_delivered = 0;
  auto source = [&num_delivered]() {
    if (num_delivered < 5) {
      return Future<TestInt>::MakeFinished(num_delivered++);
    } else {
      return Future<TestInt>::MakeFinished(IterationTraits<TestInt>::End());
    }
  };
  auto readahead = MakeReadaheadGenerator<TestInt>(source, 10);
  // Should not pump until first item requested
  ASSERT_EQ(0, num_delivered);

  auto first = readahead();
  // At this point the pumping should have happened
  ASSERT_EQ(5, num_delivered);
  ASSERT_FINISHES_OK_AND_ASSIGN(auto first_val, first);
  ASSERT_EQ(TestInt(0), first_val);

  // Read the rest
  for (int i = 0; i < 4; i++) {
    auto next = readahead();
    ASSERT_FINISHES_OK_AND_ASSIGN(auto next_val, next);
    ASSERT_EQ(TestInt(i + 1), next_val);
  }

  // Next should be end
  auto last = readahead();
  ASSERT_FINISHES_OK_AND_ASSIGN(auto last_val, last);
  ASSERT_TRUE(IsIterationEnd(last_val));
}

TEST(TestAsyncUtil, ReadaheadFailed) {
  ASSERT_OK_AND_ASSIGN(auto thread_pool, internal::ThreadPool::Make(4));
  std::atomic<int32_t> counter(0);
  // All tasks are a little slow.  The first task fails.
  // The readahead will have spawned 9 more tasks and they
  // should all pass
  auto source = [thread_pool, &counter]() -> Future<TestInt> {
    auto count = counter++;
    return *thread_pool->Submit([count]() -> Result<TestInt> {
      if (count == 0) {
        return Status::Invalid("X");
      }
      return TestInt(count);
    });
  };
  auto readahead = MakeReadaheadGenerator<TestInt>(source, 10);
  ASSERT_FINISHES_ERR(Invalid, readahead());
  SleepABit();

  for (int i = 0; i < 9; i++) {
    ASSERT_FINISHES_OK_AND_ASSIGN(auto next_val, readahead());
    ASSERT_EQ(TestInt(i + 1), next_val);
  }
  ASSERT_FINISHES_OK_AND_ASSIGN(auto after, readahead());

  // It's possible that finished was set quickly and there
  // are only 10 elements
  if (IsIterationEnd(after)) {
    return;
  }

  // It's also possible that finished was too slow and there
  // ended up being 11 elements
  ASSERT_EQ(TestInt(10), after);
  // There can't be 12 elements because SleepABit will prevent it
  ASSERT_FINISHES_OK_AND_ASSIGN(auto definitely_last, readahead());
  ASSERT_TRUE(IsIterationEnd(definitely_last));
}

TEST(TestAsyncIteratorTransform, SkipSome) {
  auto original = AsyncVectorIt<TestInt>({1, 2, 3});
  auto filter = MakeFilter([](TestInt& t) { return t.value != 2; });
  auto filtered = MakeAsyncGenerator(std::move(original), filter);
  AssertAsyncGeneratorMatch({"1", "3"}, std::move(filtered));
}

}  // namespace arrow
