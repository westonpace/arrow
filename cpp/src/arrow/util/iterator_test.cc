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
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <ostream>
#include <thread>
#include <unordered_set>
#include <vector>

#include "arrow/testing/gtest_util.h"
#include "arrow/util/async_generator.h"
#include "arrow/util/iterator.h"
#include "arrow/util/vector.h"
namespace arrow {

struct TestInt {
  TestInt() : value(-999) {}
  TestInt(int i) : value(i) {}  // NOLINT runtime/explicit
  int value;

  bool operator==(const TestInt& other) const { return value == other.value; }

  friend std::ostream& operator<<(std::ostream& os, const TestInt& v) {
    os << "{" << v.value << "}";
    return os;
  }
};

template <>
struct IterationTraits<TestInt> {
  static TestInt End() { return TestInt(); }
  static bool IsEnd(const TestInt& val) { return val == IterationTraits<TestInt>::End(); }
};

struct TestStr {
  TestStr() : value("") {}
  TestStr(const std::string& s) : value(s) {}  // NOLINT runtime/explicit
  TestStr(const char* s) : value(s) {}         // NOLINT runtime/explicit
  explicit TestStr(const TestInt& test_int) {
    if (IterationTraits<TestInt>::IsEnd(test_int)) {
      value = "";
    } else {
      value = std::to_string(test_int.value);
    }
  }
  std::string value;

  bool operator==(const TestStr& other) const { return value == other.value; }

  friend std::ostream& operator<<(std::ostream& os, const TestStr& v) {
    os << "{\"" << v.value << "\"}";
    return os;
  }
};

template <>
struct IterationTraits<TestStr> {
  static TestStr End() { return TestStr(); }
  static bool IsEnd(const TestStr& val) { return val == IterationTraits<TestStr>::End(); }
};

template <typename T>
class TracingIterator {
 public:
  explicit TracingIterator(Iterator<T> it) : it_(std::move(it)), state_(new State) {}

  Result<T> Next() {
    auto lock = state_->Lock();
    state_->thread_ids_.insert(std::this_thread::get_id());

    RETURN_NOT_OK(state_->GetNextStatus());

    ARROW_ASSIGN_OR_RAISE(auto out, it_.Next());
    state_->values_.push_back(out);

    state_->cv_.notify_one();
    return out;
  }

  class State {
   public:
    const std::vector<T>& values() { return values_; }

    const std::unordered_set<std::thread::id>& thread_ids() { return thread_ids_; }

    void InsertFailure(Status st) {
      auto lock = Lock();
      next_status_ = std::move(st);
    }

    // Wait until the iterator has emitted at least `size` values
    void WaitForValues(int size) {
      auto lock = Lock();
      cv_.wait(lock, [&]() { return values_.size() >= static_cast<size_t>(size); });
    }

    void AssertValuesEqual(const std::vector<T>& expected) {
      auto lock = Lock();
      ASSERT_EQ(values_, expected);
    }

    void AssertValuesStartwith(const std::vector<T>& expected) {
      auto lock = Lock();
      ASSERT_TRUE(std::equal(expected.begin(), expected.end(), values_.begin()));
    }

    std::unique_lock<std::mutex> Lock() { return std::unique_lock<std::mutex>(mutex_); }

   private:
    friend TracingIterator;

    Status GetNextStatus() {
      if (next_status_.ok()) {
        return Status::OK();
      }

      Status st = std::move(next_status_);
      next_status_ = Status::OK();
      return st;
    }

    Status next_status_;
    std::vector<T> values_;
    std::unordered_set<std::thread::id> thread_ids_;

    std::mutex mutex_;
    std::condition_variable cv_;
  };

  const std::shared_ptr<State>& state() const { return state_; }

 private:
  Iterator<T> it_;

  std::shared_ptr<State> state_;
};

template <typename T>
inline Iterator<T> EmptyIt() {
  return MakeEmptyIterator<T>();
}
inline Iterator<TestInt> VectorIt(std::vector<TestInt> v) {
  return MakeVectorIterator<TestInt>(std::move(v));
}

template <typename T>
AsyncGenerator<T> AsyncVectorIt(std::vector<T> v) {
  return MakeVectorGenerator(std::move(v));
}

template <typename T>
AsyncGenerator<T> Slowdown(AsyncGenerator<T> source, double seconds) {
  return MakeMappedGenerator<T, T>(
      std::move(source), [seconds](const T& res) -> Future<T> {
        return SleepAsync(seconds).Then([res](...) { return res; });
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

std::vector<TestInt> RangeVector(unsigned int max) {
  std::vector<TestInt> range(max);
  for (unsigned int i = 0; i < max; i++) {
    range[i] = i;
  }
  return range;
}

template <typename T>
inline Iterator<T> VectorIt(std::vector<T> v) {
  return MakeVectorIterator<T>(std::move(v));
}

template <typename Fn, typename T>
inline Iterator<T> FilterIt(Iterator<T> it, Fn&& fn) {
  return MakeFilterIterator(std::forward<Fn>(fn), std::move(it));
}

template <typename T>
inline Iterator<T> FlattenIt(Iterator<Iterator<T>> its) {
  return MakeFlattenIterator(std::move(its));
}

template <typename T>
void AssertIteratorMatch(std::vector<T> expected, Iterator<T> actual) {
  EXPECT_EQ(expected, IteratorToVector(std::move(actual)));
}

template <typename T>
void AssertAsyncGeneratorMatch(std::vector<T> expected, AsyncGenerator<T> actual) {
  auto vec_future = CollectAsyncGenerator(std::move(actual));
  EXPECT_OK_AND_ASSIGN(auto vec, vec_future.result());
  EXPECT_EQ(expected, vec);
}

template <typename T>
void AssertIteratorNoMatch(std::vector<T> expected, Iterator<T> actual) {
  EXPECT_NE(expected, IteratorToVector(std::move(actual)));
}

template <typename T>
void AssertIteratorNext(T expected, Iterator<T>& it) {
  ASSERT_OK_AND_ASSIGN(T actual, it.Next());
  ASSERT_EQ(expected, actual);
}

template <typename T>
void AssertIteratorExhausted(Iterator<T>& it) {
  ASSERT_OK_AND_ASSIGN(T next, it.Next());
  ASSERT_TRUE(IterationTraits<T>::IsEnd(next));
}

template <typename T>
void AssertGeneratorExhausted(AsyncGenerator<T>& gen) {
  ASSERT_FINISHES_OK_AND_ASSIGN(auto next, gen());
  ASSERT_TRUE(IterationTraits<T>::IsEnd(next));
}

// --------------------------------------------------------------------
// Synchronous iterator tests

TEST(TestEmptyIterator, Basic) { AssertIteratorMatch({}, EmptyIt<TestInt>()); }

TEST(TestVectorIterator, Basic) {
  AssertIteratorMatch({}, VectorIt({}));
  AssertIteratorMatch({1, 2, 3}, VectorIt({1, 2, 3}));

  AssertIteratorNoMatch({1}, VectorIt({}));
  AssertIteratorNoMatch({}, VectorIt({1, 2, 3}));
  AssertIteratorNoMatch({1, 2, 2}, VectorIt({1, 2, 3}));
  AssertIteratorNoMatch({1, 2, 3, 1}, VectorIt({1, 2, 3}));

  // int does not have specialized IterationTraits
  std::vector<int> elements = {0, 1, 2, 3, 4, 5};
  std::vector<int*> expected;
  for (int& element : elements) {
    expected.push_back(&element);
  }
  AssertIteratorMatch(expected, MakeVectorPointingIterator(std::move(elements)));
}

TEST(TestVectorIterator, RangeForLoop) {
  std::vector<TestInt> ints = {1, 2, 3, 4};

  auto ints_it = ints.begin();
  for (auto maybe_i : VectorIt(ints)) {
    ASSERT_OK_AND_ASSIGN(TestInt i, maybe_i);
    ASSERT_EQ(i, *ints_it++);
  }
  ASSERT_EQ(ints_it, ints.end()) << *ints_it << "@" << (ints_it - ints.begin());

  std::vector<std::unique_ptr<TestInt>> intptrs;
  for (TestInt i : ints) {
    intptrs.emplace_back(new TestInt(i));
  }

  // also works with move only types
  ints_it = ints.begin();
  for (auto maybe_i_ptr : MakeVectorIterator(std::move(intptrs))) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestInt> i_ptr, maybe_i_ptr);
    ASSERT_EQ(*i_ptr, *ints_it++);
  }
  ASSERT_EQ(ints_it, ints.end());
}

Transformer<TestInt, TestStr> MakeFirstN(int n) {
  int remaining = n;
  return [remaining](TestInt next) mutable -> Result<TransformFlow<TestStr>> {
    if (remaining > 0) {
      remaining--;
      return TransformYield(TestStr(next));
    }
    return TransformFinish();
  };
}

template <typename T>
Transformer<T, T> MakeFirstNGeneric(int n) {
  int remaining = n;
  return [remaining](T next) mutable -> Result<TransformFlow<T>> {
    if (remaining > 0) {
      remaining--;
      return TransformYield(next);
    }
    return TransformFinish();
  };
}

TEST(TestIteratorTransform, Truncating) {
  auto original = VectorIt({1, 2, 3});
  auto truncated = MakeTransformedIterator(std::move(original), MakeFirstN(2));
  AssertIteratorMatch({"1", "2"}, std::move(truncated));
}

TEST(TestIteratorTransform, TestPointer) {
  auto original = VectorIt<std::shared_ptr<int>>(
      {std::make_shared<int>(1), std::make_shared<int>(2), std::make_shared<int>(3)});
  auto truncated = MakeTransformedIterator(std::move(original),
                                           MakeFirstNGeneric<std::shared_ptr<int>>(2));
  ASSERT_OK_AND_ASSIGN(auto result, truncated.ToVector());
  ASSERT_EQ(2, result.size());
}

TEST(TestIteratorTransform, TruncatingShort) {
  // Tests the failsafe case where we never call Finish
  auto original = VectorIt({1});
  auto truncated =
      MakeTransformedIterator<TestInt, TestStr>(std::move(original), MakeFirstN(2));
  AssertIteratorMatch({"1"}, std::move(truncated));
}

Transformer<TestInt, TestStr> MakeFilter(std::function<bool(TestInt&)> filter) {
  return [filter](TestInt next) -> Result<TransformFlow<TestStr>> {
    if (filter(next)) {
      return TransformYield(TestStr(next));
    } else {
      return TransformSkip();
    }
  };
}

TEST(TestIteratorTransform, SkipSome) {
  // Exercises TransformSkip
  auto original = VectorIt({1, 2, 3});
  auto filter = MakeFilter([](TestInt& t) { return t.value != 2; });
  auto filtered = MakeTransformedIterator(std::move(original), filter);
  AssertIteratorMatch({"1", "3"}, std::move(filtered));
}

TEST(TestIteratorTransform, SkipAll) {
  // Exercises TransformSkip
  auto original = VectorIt({1, 2, 3});
  auto filter = MakeFilter([](TestInt& t) { return false; });
  auto filtered = MakeTransformedIterator(std::move(original), filter);
  AssertIteratorMatch({}, std::move(filtered));
}

Transformer<TestInt, TestStr> MakeAbortOnSecond() {
  int counter = 0;
  return [counter](TestInt next) mutable -> Result<TransformFlow<TestStr>> {
    if (counter++ == 1) {
      return Status::Invalid("X");
    }
    return TransformYield(TestStr(next));
  };
}

TEST(TestIteratorTransform, Abort) {
  auto original = VectorIt({1, 2, 3});
  auto transformed = MakeTransformedIterator(std::move(original), MakeAbortOnSecond());
  ASSERT_OK(transformed.Next());
  ASSERT_RAISES(Invalid, transformed.Next());
  ASSERT_OK_AND_ASSIGN(auto third, transformed.Next());
  ASSERT_TRUE(IterationTraits<TestStr>::IsEnd(third));
}

template <typename T>
Transformer<T, T> MakeRepeatN(int repeat_count) {
  int current_repeat = 0;
  return [repeat_count, current_repeat](T next) mutable -> Result<TransformFlow<T>> {
    current_repeat++;
    bool ready_for_next = false;
    if (current_repeat == repeat_count) {
      current_repeat = 0;
      ready_for_next = true;
    }
    return TransformYield(next, ready_for_next);
  };
}

TEST(TestIteratorTransform, Repeating) {
  auto original = VectorIt({1, 2, 3});
  auto repeated = MakeTransformedIterator<TestInt, TestInt>(std::move(original),
                                                            MakeRepeatN<TestInt>(2));
  AssertIteratorMatch({1, 1, 2, 2, 3, 3}, std::move(repeated));
}

TEST(TestFunctionIterator, RangeForLoop) {
  int i = 0;
  auto fails_at_3 = MakeFunctionIterator([&]() -> Result<TestInt> {
    if (i >= 3) {
      return Status::IndexError("fails at 3");
    }
    return i++;
  });

  int expected_i = 0;
  for (auto maybe_i : fails_at_3) {
    if (expected_i < 3) {
      ASSERT_OK(maybe_i.status());
      ASSERT_EQ(*maybe_i, expected_i);
    } else if (expected_i == 3) {
      ASSERT_RAISES(IndexError, maybe_i.status());
    }
    ASSERT_LE(expected_i, 3) << "iteration stops after an error is encountered";
    ++expected_i;
  }
}

TEST(FilterIterator, Basic) {
  AssertIteratorMatch({1, 2, 3, 4}, FilterIt(VectorIt({1, 2, 3, 4}), [](TestInt i) {
                        return FilterIterator::Accept(std::move(i));
                      }));

  AssertIteratorMatch({}, FilterIt(VectorIt({1, 2, 3, 4}), [](TestInt i) {
                        return FilterIterator::Reject<TestInt>();
                      }));

  AssertIteratorMatch({2, 4}, FilterIt(VectorIt({1, 2, 3, 4}), [](TestInt i) {
                        return i.value % 2 == 0 ? FilterIterator::Accept(std::move(i))
                                                : FilterIterator::Reject<TestInt>();
                      }));
}

TEST(FlattenVectorIterator, Basic) {
  // Flatten expects to consume Iterator<Iterator<T>>
  AssertIteratorMatch({}, FlattenIt(EmptyIt<Iterator<TestInt>>()));

  std::vector<Iterator<TestInt>> ok;
  ok.push_back(VectorIt({1}));
  ok.push_back(VectorIt({2}));
  ok.push_back(VectorIt({3}));
  AssertIteratorMatch({1, 2, 3}, FlattenIt(VectorIt(std::move(ok))));

  std::vector<Iterator<TestInt>> not_enough;
  not_enough.push_back(VectorIt({1}));
  not_enough.push_back(VectorIt({2}));
  AssertIteratorNoMatch({1, 2, 3}, FlattenIt(VectorIt(std::move(not_enough))));

  std::vector<Iterator<TestInt>> too_much;
  too_much.push_back(VectorIt({1}));
  too_much.push_back(VectorIt({2}));
  too_much.push_back(VectorIt({3}));
  too_much.push_back(VectorIt({2}));
  AssertIteratorNoMatch({1, 2, 3}, FlattenIt(VectorIt(std::move(too_much))));
}

Iterator<TestInt> Join(TestInt a, TestInt b) {
  std::vector<Iterator<TestInt>> joined{2};
  joined[0] = VectorIt({a});
  joined[1] = VectorIt({b});

  return FlattenIt(VectorIt(std::move(joined)));
}

Iterator<TestInt> Join(TestInt a, Iterator<TestInt> b) {
  std::vector<Iterator<TestInt>> joined{2};
  joined[0] = VectorIt(std::vector<TestInt>{a});
  joined[1] = std::move(b);

  return FlattenIt(VectorIt(std::move(joined)));
}

TEST(FlattenVectorIterator, Pyramid) {
  auto it = Join(1, Join(2, Join(2, Join(3, Join(3, 3)))));
  AssertIteratorMatch({1, 2, 2, 3, 3, 3}, std::move(it));
}

TEST(ReadaheadIterator, Empty) {
  ASSERT_OK_AND_ASSIGN(auto it, MakeReadaheadIterator(VectorIt({}), 2));
  AssertIteratorMatch({}, std::move(it));
}

TEST(ReadaheadIterator, Basic) {
  ASSERT_OK_AND_ASSIGN(auto it, MakeReadaheadIterator(VectorIt({1, 2, 3, 4, 5}), 2));
  AssertIteratorMatch({1, 2, 3, 4, 5}, std::move(it));
}

TEST(ReadaheadIterator, NotExhausted) {
  ASSERT_OK_AND_ASSIGN(auto it, MakeReadaheadIterator(VectorIt({1, 2, 3, 4, 5}), 2));
  AssertIteratorNext({1}, it);
  AssertIteratorNext({2}, it);
}

void SleepABit(double seconds = 1e-3) {
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
}

TEST(ReadaheadIterator, Trace) {
  TracingIterator<TestInt> tracing_it(VectorIt({1, 2, 3, 4, 5, 6, 7, 8}));
  auto tracing = tracing_it.state();
  ASSERT_EQ(tracing->values().size(), 0);

  ASSERT_OK_AND_ASSIGN(
      auto it, MakeReadaheadIterator(Iterator<TestInt>(std::move(tracing_it)), 2));
  SleepABit();  // Background iterator won't start pumping until first request comes in
  ASSERT_EQ(tracing->values().size(), 0);

  AssertIteratorNext({1}, it);  // Once we ask for one value we should get that one value
                                // as well as 2 read ahead

  tracing->WaitForValues(3);
  tracing->AssertValuesEqual({1, 2, 3});

  SleepABit();  // No further values should be fetched
  tracing->AssertValuesEqual({1, 2, 3});

  AssertIteratorNext({2}, it);
  AssertIteratorNext({3}, it);
  AssertIteratorNext({4}, it);
  tracing->WaitForValues(6);
  SleepABit();
  tracing->AssertValuesEqual({1, 2, 3, 4, 5, 6});

  AssertIteratorNext({5}, it);
  AssertIteratorNext({6}, it);
  AssertIteratorNext({7}, it);
  tracing->WaitForValues(9);
  SleepABit();
  tracing->AssertValuesEqual({1, 2, 3, 4, 5, 6, 7, 8, {}});

  AssertIteratorNext({8}, it);
  AssertIteratorExhausted(it);
  AssertIteratorExhausted(it);  // Again
  tracing->WaitForValues(9);
  SleepABit();
  tracing->AssertValuesStartwith({1, 2, 3, 4, 5, 6, 7, 8, {}});
  // A couple more EOF values may have been emitted
  const auto& values = tracing->values();
  ASSERT_LE(values.size(), 11);
  for (size_t i = 9; i < values.size(); ++i) {
    ASSERT_EQ(values[i], TestInt());
  }

  // Values were all emitted from the same thread, and it's not this thread
  const auto& thread_ids = tracing->thread_ids();
  ASSERT_EQ(thread_ids.size(), 1);
  ASSERT_NE(*thread_ids.begin(), std::this_thread::get_id());
}

TEST(ReadaheadIterator, NextError) {
  TracingIterator<TestInt> tracing_it((VectorIt({1, 2, 3})));
  auto tracing = tracing_it.state();
  ASSERT_EQ(tracing->values().size(), 0);

  tracing->InsertFailure(Status::IOError("xxx"));

  ASSERT_OK_AND_ASSIGN(
      auto it, MakeReadaheadIterator(Iterator<TestInt>(std::move(tracing_it)), 2));

  ASSERT_RAISES(IOError, it.Next().status());

  AssertIteratorExhausted(it);
  SleepABit();
  tracing->AssertValuesEqual({});
  AssertIteratorExhausted(it);
}

// --------------------------------------------------------------------
// Asynchronous iterator tests

template <typename T>
class ReentrantChecker {
 public:
  explicit ReentrantChecker(AsyncGenerator<T> source)
      : state_(std::make_shared<State>(std::move(source))) {}

  Future<T> operator()() {
    if (state_->in.load()) {
      state_->valid.store(false);
    }
    state_->in.store(true);
    auto result = state_->source();
    return result.Then(Callback{state_});
  }

  void AssertValid() {
    EXPECT_EQ(true, state_->valid.load())
        << "The generator was accessed in a reentrant manner";
  }

 private:
  struct State {
    explicit State(AsyncGenerator<T> source_)
        : source(std::move(source_)), in(false), valid(true) {}

    AsyncGenerator<T> source;
    std::atomic<bool> in;
    std::atomic<bool> valid;
  };
  struct Callback {
    Future<T> operator()(const Result<T>& result) {
      state_->in.store(false);
      return result;
    }
    std::shared_ptr<State> state_;
  };

  std::shared_ptr<State> state_;
};

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
    return SleepAsync(1e-3).Then([in](...) { return TestStr(std::to_string(in.value)); });
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
  // Mapper blocks until signal, should start multiple map tasks
  std::atomic<bool> can_proceed(false);
  std::function<Future<TestStr>(const TestInt&)> mapper =
      [&can_proceed, &map_tasks_running](const TestInt& in) {
        map_tasks_running.fetch_add(1);
        while (!can_proceed.load()) {
          SleepABit();
        }
        return TestStr(std::to_string(in.value));
      };
  auto mapped = MakeMappedGenerator(std::move(source), mapper);

  EXPECT_EQ(0, tracker.num_read());

  auto one = mapped();
  auto two = mapped();

  for (int i = 0; i < 1000; i++) {
    SleepABit();
    if (map_tasks_running.load() >= 2) {
      break;
    }
  }
  EXPECT_EQ(2, map_tasks_running.load());
  EXPECT_EQ(2, tracker.num_read());

  auto end_one = mapped();
  auto end_two = mapped();

  can_proceed.store(true);
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
    gen = ReentrantChecker<TestInt>(std::move(gen));
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

TEST(TestAsyncUtil, ConcatMap) {
  std::vector<TestInt> inputOne{1, 2, 3};
  std::vector<TestInt> inputTwo{4, 5, 6};
  std::vector<TestInt> expected{1, 2, 3, 4, 5, 6};
  auto gen = AsyncVectorIt<AsyncGenerator<TestInt>>(
      {AsyncVectorIt<TestInt>(inputOne), AsyncVectorIt<TestInt>(inputTwo)});
  auto concat = MakeConcatMapGenerator(gen);
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

TEST_P(GeneratorTestFixture, MergeMap) {
  auto gen = AsyncVectorIt<AsyncGenerator<TestInt>>(
      {MakeSource({1, 2, 3}), MakeSource({4, 5, 6})});

  auto concat_gen = MakeMergeMapGenerator(gen, 10);
  ASSERT_FINISHES_OK_AND_ASSIGN(auto concat, CollectAsyncGenerator(concat_gen));
  auto concat_ints =
      internal::MapVector([](const TestInt& val) { return val.value; }, concat);
  std::set<int> concat_set(concat_ints.begin(), concat_ints.end());

  std::set<int> expected{1, 2, 4, 3, 5, 6};
  ASSERT_EQ(expected, concat_set);
}

TEST_P(GeneratorTestFixture, MergeMapLimitedSubscriptions) {
  auto gen = AsyncVectorIt<AsyncGenerator<TestInt>>(
      {MakeSource({1, 2}), MakeSource({3, 4}), MakeSource({5, 6, 7, 8}),
       MakeSource({9, 10, 11, 12})});
  TrackingGenerator<AsyncGenerator<TestInt>> tracker(std::move(gen));
  auto merged =
      MakeMergeMapGenerator(AsyncGenerator<AsyncGenerator<TestInt>>(tracker), 2);

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

TEST_P(GeneratorTestFixture, MergeMapStress) {
  constexpr int NGENERATORS = 10;
  constexpr int NITEMS = 10;
  for (int i = 0; i < GetNumItersForStress(); i++) {
    std::vector<AsyncGenerator<TestInt>> sources;
    for (int j = 0; j < NGENERATORS; j++) {
      sources.push_back(ReentrantChecker<TestInt>(MakeSource(RangeVector(NITEMS))));
    }
    AsyncGenerator<AsyncGenerator<TestInt>> source_gen =
        ReentrantChecker<AsyncGenerator<TestInt>>(AsyncVectorIt(sources));
    auto merged = MakeMergeMapGenerator(source_gen, 4);
    ASSERT_FINISHES_OK_AND_ASSIGN(auto items, CollectAsyncGenerator(merged));
    ASSERT_EQ(NITEMS * NGENERATORS, items.size());
  }
}

TEST_P(GeneratorTestFixture, MergeMapParallelStress) {
  constexpr int NGENERATORS = 10;
  constexpr int NITEMS = 10;
  for (int i = 0; i < GetNumItersForStress(); i++) {
    std::vector<AsyncGenerator<TestInt>> sources;
    for (int j = 0; j < NGENERATORS; j++) {
      sources.push_back(MakeSource(RangeVector(NITEMS)));
    }
    auto merged = MakeMergeMapGenerator(AsyncVectorIt(sources), 4);
    merged = MakeReadaheadGenerator(merged, 4);
    ASSERT_FINISHES_OK_AND_ASSIGN(auto items, CollectAsyncGenerator(merged));
    ASSERT_EQ(NITEMS * NGENERATORS, items.size());
  }
}

INSTANTIATE_TEST_CASE_P(GeneratorTests, GeneratorTestFixture,
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
  ASSERT_TRUE(future.is_finished());
  ASSERT_OK_AND_ASSIGN(auto actual, future.result());
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
  ASSERT_TRUE(IterationTraits<TestInt>::IsEnd(one_fin));

  ASSERT_FINISHES_OK_AND_ASSIGN(auto two_fin, two);
  ASSERT_TRUE(IterationTraits<TestInt>::IsEnd(two_fin));
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
  AsyncGenerator<TestInt> it = BackgroundAsyncVectorIt({1, 2, 3, 4, 5});
  ReentrantChecker<TestInt> checker(std::move(it));
  SerialReadaheadGenerator<TestInt> serial_readahead(checker, 2);
  AssertAsyncGeneratorMatch({1, 2, 3, 4, 5},
                            static_cast<AsyncGenerator<TestInt>>(serial_readahead));
  checker.AssertValid();
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
    AsyncGenerator<TestInt> it = BackgroundAsyncVectorIt(RangeVector(NITEMS));
    ReentrantChecker<TestInt> checker(std::move(it));
    SerialReadaheadGenerator<TestInt> serial_readahead(checker, 2);
    auto visit_fut =
        VisitAsyncGenerator<TestInt>(serial_readahead, [](TestInt test_int) -> Status {
          // Normally sleeping in a visit function would be a faux-pas but we want to slow
          // the reader down to match the producer to maximize the stress
          std::this_thread::sleep_for(kYieldDuration);
          return Status::OK();
        });
    ASSERT_FINISHES_OK(visit_fut);
    checker.AssertValid();
  }
}

TEST(TestAsyncUtil, SerialReadaheadStressFast) {
  constexpr int NTASKS = 20;
  constexpr int NITEMS = 50;
  for (int i = 0; i < NTASKS; i++) {
    AsyncGenerator<TestInt> it = BackgroundAsyncVectorIt(RangeVector(NITEMS), false);
    ReentrantChecker<TestInt> checker(std::move(it));
    SerialReadaheadGenerator<TestInt> serial_readahead(checker, 2);
    auto visit_fut = VisitAsyncGenerator<TestInt>(
        serial_readahead, [](TestInt test_int) -> Status { return Status::OK(); });
    ASSERT_FINISHES_OK(visit_fut);
    checker.AssertValid();
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
  ASSERT_TRUE(IterationTraits<TestInt>::IsEnd(last_val));
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
  if (IterationTraits<TestInt>::IsEnd(after)) {
    return;
  }

  // It's also possible that finished was too slow and there
  // ended up being 11 elements
  ASSERT_EQ(TestInt(10), after);
  // There can't be 12 elements because SleepABit will prevent it
  ASSERT_FINISHES_OK_AND_ASSIGN(auto definitely_last, readahead());
  ASSERT_TRUE(IterationTraits<TestInt>::IsEnd(definitely_last));
}

TEST(TestAsyncIteratorTransform, SkipSome) {
  auto original = AsyncVectorIt<TestInt>({1, 2, 3});
  auto filter = MakeFilter([](TestInt& t) { return t.value != 2; });
  auto filtered = MakeAsyncGenerator(std::move(original), filter);
  AssertAsyncGeneratorMatch({"1", "3"}, std::move(filtered));
}

}  // namespace arrow
