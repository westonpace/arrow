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

#pragma once

#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "arrow/result.h"
#include "arrow/status.h"
#include "arrow/type_fwd.h"
#include "arrow/util/executor.h"
#include "arrow/util/functional.h"
#include "arrow/util/macros.h"
#include "arrow/util/optional.h"
#include "arrow/util/type_fwd.h"
#include "arrow/util/visibility.h"

namespace arrow {

// An object that waits on multiple futures at once.  Only one waiter
// can be registered for each future at any time.
class ARROW_EXPORT FutureWaiter {
 public:
  enum Kind : int8_t { ANY, ALL, ALL_OR_FIRST_FAILED, ITERATE };

  // HUGE_VAL isn't constexpr on Windows
  // https://social.msdn.microsoft.com/Forums/vstudio/en-US/47e8b9ff-b205-4189-968e-ee3bc3e2719f/constexpr-compile-error?forum=vclanguage
  static const double kInfinity;

  static std::unique_ptr<FutureWaiter> Make(Kind kind, std::vector<FutureImpl*> futures);

  template <typename FutureType>
  static std::unique_ptr<FutureWaiter> Make(Kind kind,
                                            const std::vector<FutureType>& futures) {
    return Make(kind, ExtractFutures(futures));
  }

  virtual ~FutureWaiter();

  bool Wait(double seconds = kInfinity);
  int WaitAndFetchOne();

  std::vector<int> MoveFinishedFutures();

 protected:
  // Extract FutureImpls from Futures
  template <typename FutureType,
            typename Enable = std::enable_if<!std::is_pointer<FutureType>::value>>
  static std::vector<FutureImpl*> ExtractFutures(const std::vector<FutureType>& futures) {
    std::vector<FutureImpl*> base_futures(futures.size());
    for (int i = 0; i < static_cast<int>(futures.size()); ++i) {
      base_futures[i] = futures[i].impl_.get();
    }
    return base_futures;
  }

  // Extract FutureImpls from Future pointers
  template <typename FutureType>
  static std::vector<FutureImpl*> ExtractFutures(
      const std::vector<FutureType*>& futures) {
    std::vector<FutureImpl*> base_futures(futures.size());
    for (int i = 0; i < static_cast<int>(futures.size()); ++i) {
      base_futures[i] = futures[i]->impl_.get();
    }
    return base_futures;
  }

  FutureWaiter();
  ARROW_DISALLOW_COPY_AND_ASSIGN(FutureWaiter);

  inline void MarkFutureFinishedUnlocked(int future_num, FutureState state);

  friend class FutureImpl;
  friend class ConcreteFutureImpl;
};

template <typename T>
Future<T>::operator Future<>() const {
  Future<> status_future;
  status_future.impl_ = FutureBase<T>::impl_;
  return status_future;
}

/// If a Result<Future> holds an error instead of a Future, construct a finished Future
/// holding that error.
template <typename T>
static Future<T> DeferNotOk(Result<Future<T>> maybe_future) {
  if (ARROW_PREDICT_FALSE(!maybe_future.ok())) {
    return Future<T>::MakeFinished(std::move(maybe_future).status());
  }
  return std::move(maybe_future).MoveValueUnsafe();
}

/// \brief Wait for all the futures to end, or for the given timeout to expire.
///
/// `true` is returned if all the futures completed before the timeout was reached,
/// `false` otherwise.
template <typename T>
inline bool WaitForAll(const std::vector<Future<T>>& futures,
                       double seconds = FutureWaiter::kInfinity) {
  auto waiter = FutureWaiter::Make(FutureWaiter::ALL, futures);
  return waiter->Wait(seconds);
}

/// \brief Wait for all the futures to end, or for the given timeout to expire.
///
/// `true` is returned if all the futures completed before the timeout was reached,
/// `false` otherwise.
template <typename T>
inline bool WaitForAll(const std::vector<Future<T>*>& futures,
                       double seconds = FutureWaiter::kInfinity) {
  auto waiter = FutureWaiter::Make(FutureWaiter::ALL, futures);
  return waiter->Wait(seconds);
}

/// \brief Create a Future which completes when all of `futures` complete.
///
/// The future's result is a vector of the results of `futures`.
/// Note that this future will never be marked "failed"; failed results
/// will be stored in the result vector alongside successful results.
template <typename T>
Future<std::vector<Result<T>>> All(std::vector<Future<T>> futures) {
  struct State {
    explicit State(std::vector<Future<T>> f)
        : futures(std::move(f)), n_remaining(futures.size()) {}

    std::vector<Future<T>> futures;
    std::atomic<size_t> n_remaining;
  };

  if (futures.size() == 0) {
    return {std::vector<Result<T>>{}};
  }

  auto state = std::make_shared<State>(std::move(futures));

  auto out = Future<std::vector<Result<T>>>::Make();
  for (const Future<T>& future : state->futures) {
    future.AddCallback([state, out](const Result<T>&) mutable {
      if (state->n_remaining.fetch_sub(1) != 1) return;

      std::vector<Result<T>> results(state->futures.size());
      for (size_t i = 0; i < results.size(); ++i) {
        results[i] = state->futures[i].result();
      }
      out.MarkFinished(std::move(results));
    });
  }
  return out;
}

/// \brief Create a Future which completes when all of `futures` complete.
///
/// The future will be marked complete if all `futures` complete
/// successfully. Otherwise, it will be marked failed with the status of
/// the first failing future.
ARROW_EXPORT
Future<> AllComplete(const std::vector<Future<>>& futures);

/// \brief Wait for one of the futures to end, or for the given timeout to expire.
///
/// The indices of all completed futures are returned.  Note that some futures
/// may not be in the returned set, but still complete concurrently.
template <typename T>
inline std::vector<int> WaitForAny(const std::vector<Future<T>>& futures,
                                   double seconds = FutureWaiter::kInfinity) {
  auto waiter = FutureWaiter::Make(FutureWaiter::ANY, futures);
  waiter->Wait(seconds);
  return waiter->MoveFinishedFutures();
}

/// \brief Wait for one of the futures to end, or for the given timeout to expire.
///
/// The indices of all completed futures are returned.  Note that some futures
/// may not be in the returned set, but still complete concurrently.
template <typename T>
inline std::vector<int> WaitForAny(const std::vector<Future<T>*>& futures,
                                   double seconds = FutureWaiter::kInfinity) {
  auto waiter = FutureWaiter::Make(FutureWaiter::ANY, futures);
  waiter->Wait(seconds);
  return waiter->MoveFinishedFutures();
}

struct Continue {
  template <typename T>
  operator util::optional<T>() && {  // NOLINT explicit
    return {};
  }
};

template <typename T = internal::Empty>
util::optional<T> Break(T break_value = {}) {
  return util::optional<T>{std::move(break_value)};
}

template <typename T = internal::Empty>
using ControlFlow = util::optional<T>;

template <typename T>
void ForwardControlResult(const Result<ControlFlow<T>>& result, Future<T> sink) {
  sink.MarkFinished(**result);
}
template <>
inline void ForwardControlResult(const Result<ControlFlow<>>& result, Future<> sink) {
  sink.MarkFinished();
}

/// \brief Loop through an asynchronous sequence
///
/// \param[in] iterate A generator of Future<ControlFlow<BreakValue>>. On completion
/// of each yielded future the resulting ControlFlow will be examined. A Break will
/// terminate the loop, while a Continue will re-invoke `iterate`. \return A future
/// which will complete when a Future returned by iterate completes with a Break
template <typename Iterate,
          typename Control = typename detail::result_of_t<Iterate()>::ValueType,
          typename BreakValueType = typename Control::value_type>
Future<BreakValueType> Loop(Iterate iterate) {
  struct Callback {
    bool CheckForTermination(const Result<Control>& control_res) {
      if (!control_res.ok()) {
        break_fut.MarkFinished(control_res.status());
        return true;
      }
      if (control_res->has_value()) {
        ForwardControlResult(control_res, std::move(break_fut));
        return true;
      }
      return false;
    }

    void operator()(const Result<Control>& maybe_control) && {
      if (CheckForTermination(maybe_control)) return;

      auto control_fut = iterate();
      while (true) {
        if (control_fut.TryAddCallback([this]() { return *this; })) {
          // Adding a callback succeeded; control_fut was not finished
          // and we must wait to CheckForTermination.
          return;
        }
        // Adding a callback failed; control_fut was finished and we
        // can CheckForTermination immediately. This also avoids recursion and potential
        // stack overflow.
        if (CheckForTermination(control_fut.result())) return;

        control_fut = iterate();
      }
    }

    Iterate iterate;

    // If the future returned by control_fut is never completed then we will be hanging on
    // to break_fut forever even if the listener has given up listening on it.  Instead we
    // rely on the fact that a producer (the caller of Future<>::Make) is always
    // responsible for completing the futures they create.
    // TODO: Could avoid this kind of situation with "future abandonment" similar to mesos
    Future<BreakValueType> break_fut;
  };

  auto break_fut = Future<BreakValueType>::Make();
  auto control_fut = iterate();
  control_fut.AddCallback(Callback{std::move(iterate), break_fut});

  return break_fut;
}

}  // namespace arrow
