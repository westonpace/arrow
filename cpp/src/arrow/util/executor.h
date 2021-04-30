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
#include <cstdint>

#include "arrow/result.h"
#include "arrow/util/cancel.h"
#include "arrow/util/functional.h"
#include "arrow/util/type_fwd.h"

namespace arrow {

/// \brief Describes whether the callback should be scheduled or run synchronously
enum ShouldSchedule {
  /// Always run the callback synchronously (the default)
  NEVER = 0,
  /// Schedule a new task only if the future is not finished when the
  /// callback is added
  IF_UNFINISHED = 1,
  /// Schedule a new task if the future is not finished or if there is
  /// a free thread in the executor
  IF_IDLE = 2,
  /// Always schedule the callback as a new task
  ALWAYS = 3
};

/// \brief Options that control how a continuation is run
struct CallbackOptions {
  /// Describes whether the callback should be run synchronously or scheduled
  ShouldSchedule should_schedule = ShouldSchedule::NEVER;
  /// If the callback is scheduled then this is the executor it should be scheduled
  /// on.  If this is NULL then should_schedule must be NEVER
  internal::Executor* executor = NULL;

  static CallbackOptions Defaults() { return CallbackOptions(); }
};

/// A Future's execution or completion status
enum class FutureState : int8_t { PENDING, SUCCESS, FAILURE };

template <typename Source, typename Dest>
typename std::enable_if<Source::is_empty>::type Propagate(Source& source, Dest dest) {
  struct MarkNextFinished {
    void operator()(const Status& status) && { next.MarkFinished(status); }
    Dest next;
  };
  source.AddCallback(MarkNextFinished{std::move(dest)});
}

namespace detail {

template <typename>
struct is_future : std::false_type {};

template <typename T>
struct is_future<Future<T>> : std::true_type {};

template <typename Signature>
using result_of_t = typename std::result_of<Signature>::type;

template <typename Source, typename Dest>
typename std::enable_if<Source::is_empty>::type Propagate(Source* source, Dest dest) {
  struct MarkNextFinished {
    void operator()(const Status& status) && { next.MarkFinished(status); }
    Dest next;
  };
  source->AddCallback(MarkNextFinished{std::move(dest)});
}

template <typename Source, typename Dest, bool SourceEmpty = Source::is_empty,
          bool DestEmpty = Dest::is_empty>
struct MarkNextFinished {};

template <typename Source, typename Dest>
struct MarkNextFinished<Source, Dest, true, false> {
  void operator()(const Status& status) && { next.MarkFinished(status); }
  Dest next;
};

template <typename Source, typename Dest>
struct MarkNextFinished<Source, Dest, true, true> {
  void operator()(const Status& status) && { next.MarkFinished(status); }
  Dest next;
};

template <typename Source, typename Dest>
struct MarkNextFinished<Source, Dest, false, true> {
  void operator()(const Result<typename Source::ValueType>& res) && {
    next.MarkFinished(res.status());
  }
  Dest next;
};

template <typename Source, typename Dest>
struct MarkNextFinished<Source, Dest, false, false> {
  void operator()(const Result<typename Source::ValueType>& res) && {
    next.MarkFinished(res);
  }
  Dest next;
};

struct ContinueFuture {
  template <typename Return>
  struct ForReturnImpl;

  template <typename Return>
  using ForReturn = typename ForReturnImpl<Return>::type;

  template <typename Signature>
  using ForSignature = ForReturn<result_of_t<Signature>>;

  template <typename ContinueFunc, typename... Args,
            typename ContinueResult = result_of_t<ContinueFunc && (Args && ...)>,
            typename NextFuture = ForReturn<ContinueResult>>
  typename std::enable_if<std::is_void<ContinueResult>::value>::type operator()(
      NextFuture next, ContinueFunc&& f, Args&&... a) const {
    std::forward<ContinueFunc>(f)(std::forward<Args>(a)...);
    next.MarkFinished();
  }

  template <typename ContinueFunc, typename... Args,
            typename ContinueResult = result_of_t<ContinueFunc && (Args && ...)>,
            typename NextFuture = ForReturn<ContinueResult>>
  typename std::enable_if<
      !std::is_void<ContinueResult>::value && !is_future<ContinueResult>::value &&
      (!NextFuture::is_empty || std::is_same<ContinueResult, Status>::value)>::type
  operator()(NextFuture next, ContinueFunc&& f, Args&&... a) const {
    next.MarkFinished(std::forward<ContinueFunc>(f)(std::forward<Args>(a)...));
  }

  template <typename ContinueFunc, typename... Args,
            typename ContinueResult = result_of_t<ContinueFunc && (Args && ...)>,
            typename NextFuture = ForReturn<ContinueResult>>
  typename std::enable_if<!std::is_void<ContinueResult>::value &&
                          !is_future<ContinueResult>::value && NextFuture::is_empty &&
                          !std::is_same<ContinueResult, Status>::value>::type
  operator()(NextFuture next, ContinueFunc&& f, Args&&... a) const {
    next.MarkFinished(std::forward<ContinueFunc>(f)(std::forward<Args>(a)...).status());
  }

  template <typename ContinueFunc, typename... Args,
            typename ContinueResult = result_of_t<ContinueFunc && (Args && ...)>,
            typename NextFuture = ForReturn<ContinueResult>>
  typename std::enable_if<is_future<ContinueResult>::value>::type operator()(
      NextFuture next, ContinueFunc&& f, Args&&... a) const {
    ContinueResult signal_to_complete_next =
        std::forward<ContinueFunc>(f)(std::forward<Args>(a)...);
    MarkNextFinished<ContinueResult, NextFuture> callback{std::move(next)};
    signal_to_complete_next.AddCallback(std::move(callback));
  }
};

template <>
struct ContinueFuture::ForReturnImpl<void> {
  using type = Future<>;
};

template <>
struct ContinueFuture::ForReturnImpl<Status> {
  using type = Future<>;
};

template <typename R>
struct ContinueFuture::ForReturnImpl {
  using type = Future<R>;
};

template <typename T>
struct ContinueFuture::ForReturnImpl<Result<T>> {
  using type = Future<T>;
};

template <typename T>
struct ContinueFuture::ForReturnImpl<Future<T>> {
  using type = Future<T>;
};

}  // namespace detail

inline bool IsFutureFinished(FutureState state) { return state != FutureState::PENDING; }

// Untyped private implementation
class ARROW_EXPORT FutureImpl {
 public:
  FutureImpl();
  virtual ~FutureImpl() = default;

  FutureState state() { return state_.load(); }

  static std::unique_ptr<FutureImpl> Make();
  static std::unique_ptr<FutureImpl> MakeFinished(FutureState state);

  // Future API
  void MarkFinished();
  void MarkFailed();
  void Wait();
  bool Wait(double seconds);

  using Callback = internal::FnOnce<void()>;
  void AddCallback(Callback callback, CallbackOptions opts);
  bool TryAddCallback(const std::function<Callback()>& callback_factory,
                      CallbackOptions opts);

  // Waiter API
  inline FutureState SetWaiter(FutureWaiter* w, int future_num);
  inline void RemoveWaiter(FutureWaiter* w);

  std::atomic<FutureState> state_{FutureState::PENDING};

  // Type erased storage for arbitrary results
  // XXX small objects could be stored inline instead of boxed in a pointer
  using Storage = std::unique_ptr<void, void (*)(void*)>;
  Storage result_{NULLPTR, NULLPTR};

  struct CallbackRecord {
    Callback callback;
    CallbackOptions options;
  };
  std::vector<CallbackRecord> callbacks_;
};

/// \brief EXPERIMENTAL A std::future-like class with more functionality.
///
/// A Future represents the results of a past or future computation.
/// The Future API has two sides: a producer side and a consumer side.
///
/// The producer API allows creating a Future and setting its result or
/// status, possibly after running a computation function.
///
/// The consumer API allows querying a Future's current state, wait for it
/// to complete, or wait on multiple Futures at once (using WaitForAll,
/// WaitForAny or AsCompletedIterator).
template <typename T>
class FutureBase {
 public:
  using ValueType = T;
  // The default constructor creates an invalid Future.  Use Future::Make()
  // for a valid Future.  This constructor is mostly for the convenience
  // of being able to presize a vector of Futures.
  FutureBase() = default;

  // Consumer API

  bool is_valid() const { return impl_ != NULLPTR; }

  /// \brief Return the Future's current state
  ///
  /// A return value of PENDING is only indicative, as the Future can complete
  /// concurrently.  A return value of FAILURE or SUCCESS is definitive, though.
  FutureState state() const {
    CheckValid();
    return impl_->state();
  }

  /// \brief Whether the Future is finished
  ///
  /// A false return value is only indicative, as the Future can complete
  /// concurrently.  A true return value is definitive, though.
  bool is_finished() const {
    CheckValid();
    return IsFutureFinished(impl_->state());
  }

  /// \brief Wait for the Future to complete and return its Status
  const Status& status() const {
    Wait();
    return GetResult()->status();
  }

  /// \brief Wait for the Future to complete and return its Result
  const Result<ValueType>& result() const& {
    Wait();
    return *GetResult();
  }

  /// \brief Wait for the Future to complete
  void Wait() const {
    CheckValid();
    if (!IsFutureFinished(impl_->state())) {
      impl_->Wait();
    }
  }

  /// \brief Wait for the Future to complete, or for the timeout to expire
  ///
  /// `true` is returned if the Future completed, `false` if the timeout expired.
  /// Note a `false` value is only indicative, as the Future can complete
  /// concurrently.
  bool Wait(double seconds) const {
    CheckValid();
    if (IsFutureFinished(impl_->state())) {
      return true;
    }
    return impl_->Wait(seconds);
  }

 protected:
  void InitializeFromResult(Result<ValueType> res) {
    if (ARROW_PREDICT_TRUE(res.ok())) {
      impl_ = FutureImpl::MakeFinished(FutureState::SUCCESS);
    } else {
      impl_ = FutureImpl::MakeFinished(FutureState::FAILURE);
    }
    SetResult(std::move(res));
  }

  void Initialize() { impl_ = FutureImpl::Make(); }

  Result<ValueType>* GetResult() const {
    return static_cast<Result<ValueType>*>(impl_->result_.get());
  }

  void SetResult(Result<ValueType> res) {
    impl_->result_ = {new Result<ValueType>(std::move(res)),
                      [](void* p) { delete static_cast<Result<ValueType>*>(p); }};
  }

  void DoMarkFinished(Result<ValueType> res) {
    SetResult(std::move(res));

    if (ARROW_PREDICT_TRUE(GetResult()->ok())) {
      impl_->MarkFinished();
    } else {
      impl_->MarkFailed();
    }
  }

  void CheckValid() const {
#ifndef NDEBUG
    if (!is_valid()) {
      Status::Invalid("Invalid Future (default-initialized?)").Abort();
    }
#endif
  }

  explicit FutureBase(std::shared_ptr<FutureImpl> impl) : impl_(std::move(impl)) {}

  std::shared_ptr<FutureImpl> impl_;

  friend class FutureWaiter;
  friend struct detail::ContinueFuture;
};

template <typename T>
class ARROW_MUST_USE_TYPE Future : public FutureBase<T> {
 public:
  using ValueType = T;
  using SyncType = Result<T>;
  static constexpr bool is_empty = false;

  Future() = default;

  /// \brief Returns an rvalue to the result.  This method is potentially unsafe
  ///
  /// The future is not the unique owner of the result, copies of a future will
  /// also point to the same result.  You must make sure that no other copies
  /// of the future exist.  Attempts to add callbacks after you move the result
  /// will result in undefined behavior.
  Result<ValueType>&& MoveResult() {
    FutureBase<T>::Wait();
    return std::move(*FutureBase<T>::GetResult());
  }

  /// \brief Wait for the Future to complete and return its Result (or Status for an empty
  /// future)
  ///
  /// This method is useful for general purpose code converting from async to sync where T
  /// is a template parameter and may be empty.
  const SyncType& to_sync() const { return FutureBase<T>::result(); }

  // Producer API

  /// \brief Producer API: mark Future finished
  ///
  /// The Future's result is set to `res`.
  void MarkFinished(Result<T> res) { FutureBase<T>::DoMarkFinished(std::move(res)); }

  /// \brief Producer API: instantiate a finished Future
  static Future MakeFinished(Result<T> res) {
    Future fut;
    fut.InitializeFromResult(std::move(res));
    return fut;
  }

  /// \brief Consumer API: Register a callback to run when this future completes
  ///
  /// The callback should receive the result of the future (const Result<T>&)
  /// For a void or statusy future the callback should receive the status
  ///
  /// There is no guarantee to the order in which callbacks will run.  In
  /// particular, callbacks added while the future is being marked complete
  /// may be executed immediately, ahead of, or even the same time as, other
  /// callbacks that have been previously added.
  ///
  /// By default callbacks will run synchronously as part of MarkFinished (if
  /// the callback was added before completion) or AddCallback (if the callback
  /// was added after completion)
  ///
  /// WARNING: callbacks may hold arbitrary references, including cyclic references.
  /// Since callbacks will only be destroyed after they are invoked, this can lead to
  /// memory leaks if a Future is never marked finished (abandoned):
  ///
  /// {
  ///     auto fut = Future<>::Make();
  ///     fut.AddCallback([fut](...) {});
  /// }
  ///
  /// In this example `fut` falls out of scope but is not destroyed because it holds a
  /// cyclic reference to itself through the callback.
  template <typename OnComplete>
  void AddCallback(OnComplete on_complete,
                   CallbackOptions opts = CallbackOptions::Defaults()) const {
    // We know impl_ will not be dangling when invoking callbacks because at least one
    // thread will be waiting for MarkFinished to return. Thus it's safe to keep a
    // weak reference to impl_ here
    FutureBase<T>::impl_->AddCallback(
        Callback<OnComplete>{WeakFuture<T>(*this), std::move(on_complete)}, opts);
  }

  /// \brief Overload of AddCallback that will return false instead of running
  /// synchronously
  ///
  /// This overload will guarantee the callback is never run synchronously.  If the future
  /// is already finished then it will simply return false.  This can be useful to avoid
  /// stack overflow in a situation where you have recursive Futures.  For an example
  /// see the Loop function
  ///
  /// Takes in a callback factory function to allow moving callbacks (the factory function
  /// will only be called if the callback can successfully be added)
  ///
  /// Returns true if a callback was actually added and false if the callback failed
  /// to add because the future was marked complete.
  template <typename CallbackFactory>
  bool TryAddCallback(const CallbackFactory& callback_factory,
                      CallbackOptions opts = CallbackOptions::Defaults()) const {
    return FutureBase<T>::impl_->TryAddCallback(
        [this, &callback_factory]() {
          return Callback<detail::result_of_t<CallbackFactory()>>{WeakFuture<T>(*this),
                                                                  callback_factory()};
        },
        opts);
  }

  /// \brief Consumer API: Register a continuation to run when this future completes
  ///
  /// The continuation will run in the same thread that called MarkFinished (whatever
  /// callback is registered with this function will run before MarkFinished returns).
  /// Avoid long-running callbacks in favor of submitting a task to an Executor and
  /// returning the future.
  ///
  /// Two callbacks are supported:
  /// - OnSuccess, called with the result (const ValueType&) on successul completion.
  ///              for an empty future this will be called with nothing ()
  /// - OnFailure, called with the error (const Status&) on failed completion.
  ///
  /// Then() returns a Future whose ValueType is derived from the return type of the
  /// callbacks. If a callback returns:
  /// - void, a Future<> will be returned which will completes successully as soon
  ///   as the callback runs.
  /// - Status, a Future<> will be returned which will complete with the returned Status
  ///   as soon as the callback runs.
  /// - V or Result<V>, a Future<V> will be returned which will complete with the result
  ///   of invoking the callback as soon as the callback runs.
  /// - Future<V>, a Future<V> will be returned which will be marked complete when the
  ///   future returned by the callback completes (and will complete with the same
  ///   result).
  ///
  /// The continued Future type must be the same for both callbacks.
  ///
  /// Note that OnFailure can swallow errors, allowing continued Futures to successully
  /// complete even if this Future fails.
  ///
  /// If this future is already completed then the callback will be run immediately
  /// and the returned future may already be marked complete.
  ///
  /// See AddCallback for general considerations when writing callbacks.
  template <typename OnSuccess, typename OnFailure,
            typename ContinuedFuture =
                detail::ContinueFuture::ForSignature<OnSuccess && (const T&)>>
  ContinuedFuture Then(OnSuccess on_success, OnFailure on_failure) const {
    static_assert(
        std::is_same<detail::ContinueFuture::ForSignature<OnFailure && (const Status&)>,
                     ContinuedFuture>::value,
        "OnSuccess and OnFailure must continue with the same future type");
    using OnSuccessArg =
        typename std::decay<internal::call_traits::argument_type<0, OnSuccess>>::type;
    static_assert(
        !std::is_same<OnSuccessArg, typename EnsureResult<OnSuccessArg>::type>::value,
        "OnSuccess' argument should not be a Result");

    auto next = ContinuedFuture::Make();

    struct Callback {
      void operator()(const Result<T>& result) && {
        detail::ContinueFuture continue_future;
        if (ARROW_PREDICT_TRUE(result.ok())) {
          // move on_failure to a(n immediately destroyed) temporary to free its resources
          ARROW_UNUSED(OnFailure(std::move(on_failure)));
          continue_future(std::move(next), std::move(on_success), result.ValueOrDie());
        } else {
          ARROW_UNUSED(OnSuccess(std::move(on_success)));
          continue_future(std::move(next), std::move(on_failure), result.status());
        }
      }

      OnSuccess on_success;
      OnFailure on_failure;
      ContinuedFuture next;
    };

    AddCallback(Callback{std::forward<OnSuccess>(on_success),
                         std::forward<OnFailure>(on_failure), next});

    return next;
  }

  /// \brief Overload without OnFailure. Failures will be passed through unchanged.
  ///        T value passed to callback as const ref.  Not valid for Future<>
  template <typename OnSuccess,
            typename ContinuedFuture =
                detail::ContinueFuture::ForSignature<OnSuccess && (const T&)>,
            typename E = ValueType>
  ContinuedFuture Then(OnSuccess&& on_success) const {
    return Then(std::forward<OnSuccess>(on_success), [](const Status& s) {
      return Result<typename ContinuedFuture::ValueType>(s);
    });
  }

  /// \brief Producer API: instantiate a valid Future
  ///
  /// The Future's state is initialized with PENDING.  If you are creating a future with
  /// this method you must ensure that future is eventually completed (with success or
  /// failure).  Creating a future, returning it, and never completing the future can lead
  /// to memory leaks (for example, see Loop).
  static Future Make() {
    Future fut;
    fut.impl_ = FutureImpl::Make();
    return fut;
  }

  /// \brief Future<T> is convertible to Future<>, which views only the
  /// Status of the original. Marking the returned Future Finished is not supported.
  explicit operator Future<>() const;

  /// \brief Implicit constructor to create a finished future from a value
  Future(ValueType val) : Future() {  // NOLINT runtime/explicit
    FutureBase<T>::impl_ = FutureImpl::MakeFinished(FutureState::SUCCESS);
    FutureBase<T>::SetResult(std::move(val));
  }

  /// \brief Implicit constructor to create a future from a Result, enabling use
  ///     of macros like ARROW_ASSIGN_OR_RAISE.
  Future(Result<ValueType> res) : Future() {  // NOLINT runtime/explicit
    FutureBase<T>::InitializeFromResult(std::move(res));
  }

  /// \brief Implicit constructor to create a future from a Status, enabling use
  ///     of macros like ARROW_RETURN_NOT_OK.
  Future(Status s)  // NOLINT runtime/explicit
      : Future(Result<ValueType>(std::move(s))) {}

 protected:
  // A callable object that forwards a future's result to a user-defined callback
  template <typename OnComplete>
  struct Callback {
    void operator()() && {
      auto self = weak_self.get();
      std::move(on_complete)(*self.GetResult());
    }

    WeakFuture<T> weak_self;
    OnComplete on_complete;
  };

  explicit Future(std::shared_ptr<FutureImpl> impl) : FutureBase<T>(std::move(impl)) {}

  template <typename U>
  friend class Future;
  friend class WeakFuture<T>;

  FRIEND_TEST(FutureRefTest, ChainRemoved);
  FRIEND_TEST(FutureRefTest, TailRemoved);
  FRIEND_TEST(FutureRefTest, HeadRemoved);
};

template <typename T>
class WeakFuture {
 public:
  explicit WeakFuture(const Future<T>& future) : impl_(future.impl_) {}

  Future<T> get() { return Future<T>{impl_.lock()}; }

 private:
  std::weak_ptr<FutureImpl> impl_;
};

template <>
class ARROW_MUST_USE_TYPE Future<arrow::internal::Empty>
    : public FutureBase<arrow::internal::Empty> {
 public:
  using ValueType = arrow::internal::Empty;
  using SyncType = Status;
  static constexpr bool is_empty = true;

  Future() = default;

  void MarkFinished(Status s = Status::OK()) {
    return DoMarkFinished(internal::Empty::ToResult(std::move(s)));
  }

  static Future<> MakeFinished(Status s = Status::OK()) {
    Future fut;
    fut.InitializeFromResult(internal::Empty::ToResult(std::move(s)));
    return fut;
  }

  const SyncType& to_sync() const { return status(); }

  template <typename OnComplete>
  void AddCallback(OnComplete on_complete,
                   CallbackOptions opts = CallbackOptions::Defaults()) const {
    // We know impl_ will not be dangling when invoking callbacks because at least one
    // thread will be waiting for MarkFinished to return. Thus it's safe to keep a
    // weak reference to impl_ here
    impl_->AddCallback(Callback<OnComplete>{WeakFuture<>(*this), std::move(on_complete)},
                       opts);
  }

  template <typename CallbackFactory>
  bool TryAddCallback(const CallbackFactory& callback_factory,
                      CallbackOptions opts = CallbackOptions::Defaults()) const {
    return impl_->TryAddCallback(
        [this, &callback_factory]() {
          return Callback<detail::result_of_t<CallbackFactory()>>{WeakFuture<>(*this),
                                                                  callback_factory()};
        },
        opts);
  }

  template <typename OnSuccess, typename OnFailure,
            typename ContinuedFuture =
                detail::ContinueFuture::ForSignature<OnSuccess && (void)>>
  ContinuedFuture Then(OnSuccess on_success, OnFailure on_failure) const {
    static_assert(
        std::is_same<detail::ContinueFuture::ForSignature<OnFailure && (const Status&)>,
                     ContinuedFuture>::value,
        "OnSuccess and OnFailure must continue with the same future type");

    auto next = ContinuedFuture::Make();

    struct Callback {
      void operator()(const Status& status) && {
        detail::ContinueFuture continue_future;
        if (ARROW_PREDICT_TRUE(status.ok())) {
          // move on_failure to a(n immediately destroyed) temporary to free its resources
          ARROW_UNUSED(OnFailure(std::move(on_failure)));
          continue_future(std::move(next), std::move(on_success));
        } else {
          ARROW_UNUSED(OnSuccess(std::move(on_success)));
          continue_future(std::move(next), std::move(on_failure), status);
        }
      }

      OnSuccess on_success;
      OnFailure on_failure;
      ContinuedFuture next;
    };

    AddCallback(Callback{std::forward<OnSuccess>(on_success),
                         std::forward<OnFailure>(on_failure), next});

    return next;
  }

  /// \brief Overload without OnFailure. Failures will be passed through unchanged.
  template <typename OnSuccess, typename ContinuedFuture =
                                    detail::ContinueFuture::ForSignature<OnSuccess && ()>>
  ContinuedFuture Then(OnSuccess&& on_success) const {
    return Then(std::forward<OnSuccess>(on_success), [](const Status& s) {
      return Result<typename ContinuedFuture::ValueType>(s);
    });
  }

  /// \brief Producer API: instantiate a valid Future
  ///
  /// The Future's state is initialized with PENDING.  If you are creating a future with
  /// this method you must ensure that future is eventually completed (with success or
  /// failure).  Creating a future, returning it, and never completing the future can lead
  /// to memory leaks (for example, see Loop).
  static Future Make() {
    Future fut;
    fut.Initialize();
    return fut;
  }

  /// \brief Implicit constructor to create a future from a Result, enabling use
  ///     of macros like ARROW_ASSIGN_OR_RAISE.
  Future(Result<ValueType> res) : Future() {  // NOLINT runtime/explicit
    InitializeFromResult(std::move(res));
  }

  /// \brief Implicit constructor to create a future from a Status, enabling use
  ///     of macros like ARROW_RETURN_NOT_OK.
  Future(Status s)  // NOLINT runtime/explicit
      : Future(Result<ValueType>(std::move(s))) {}

 protected:
  // A callable object that forwards a future's result to a user-defined callback
  template <typename OnComplete>
  struct Callback {
    void operator()() && {
      auto self = weak_self.get();
      std::move(on_complete)(self.GetResult()->status());
    }

    WeakFuture<> weak_self;
    OnComplete on_complete;
  };

  explicit Future(std::shared_ptr<FutureImpl> impl) : FutureBase(std::move(impl)) {}

  template <typename U>
  friend class Future;
  friend class WeakFuture<>;

  FRIEND_TEST(FutureRefTest, ChainRemoved);
  FRIEND_TEST(FutureRefTest, TailRemoved);
  FRIEND_TEST(FutureRefTest, HeadRemoved);
};

namespace internal {
// Hints about a task that may be used by an Executor.
// They are ignored by the provided ThreadPool implementation.
struct TaskHints {
  // The lower, the more urgent
  int32_t priority = 0;
  // The IO transfer size in bytes
  int64_t io_size = -1;
  // The approximate CPU cost in number of instructions
  int64_t cpu_cost = -1;
  // An application-specific ID
  int64_t external_id = -1;
};

class ARROW_EXPORT Executor {
 public:
  using StopCallback = internal::FnOnce<void(const Status&)>;

  virtual ~Executor();

  // Spawn a fire-and-forget task.
  template <typename Function>
  Status Spawn(Function&& func, StopToken stop_token = StopToken::Unstoppable()) {
    return SpawnReal(TaskHints{}, std::forward<Function>(func), std::move(stop_token),
                     StopCallback{});
  }

  template <typename Function>
  Status Spawn(TaskHints hints, Function&& func,
               StopToken stop_token = StopToken::Unstoppable()) {
    return SpawnReal(hints, std::forward<Function>(func), std::move(stop_token),
                     StopCallback{});
  }

  /// \brief Transfers a future to this executor.
  ///
  /// Any continuations added to the returned future will run in this executor.  Otherwise
  /// they would run on the same thread that called MarkFinished.
  ///
  /// This is necessary when (for example) an I/O task is completing a future.
  /// The continuations of that future should run on the CPU thread pool keeping
  /// CPU heavy work off the I/O thread pool.  So the I/O task should transfer
  /// the future to the CPU executor before returning.
  ///
  /// By default, if the future is already finished, then no transfer will take place
  /// since the callbacks will run synchronously and presumably the thread calling
  /// AddCallback/Then is from the intended executor.  This behavior can be overridden
  /// by supplying a custom should_scheulde (ShouldSchedule::NEVER is not valid)
  template <typename T, typename FT = Future<T>, typename FTSync = typename FT::SyncType>
  Future<T> Transfer(Future<T> future,
                     ShouldSchedule should_schedule = ShouldSchedule::IF_UNFINISHED) {
    if (should_schedule == ShouldSchedule::NEVER) {
      return Future<T>::MakeFinished(Status::Invalid(
          "ShouldSchedule::NEVER is not a valid optionf or Executor::Transfer"));
    }
    auto transferred = Future<T>::Make();
    if (should_schedule != ShouldSchedule::IF_UNFINISHED) {
      CallbackOptions callback_options = CallbackOptions::Defaults();
      callback_options.should_schedule = should_schedule;
      callback_options.executor = this;
      auto sync_callback = [transferred](const FTSync& result) mutable {
        transferred.MarkFinished(result);
      };
      future.AddCallback(sync_callback, callback_options);
      return transferred;
    }

    auto callback = [this, transferred](const FTSync& result) mutable {
      auto spawn_status =
          Spawn([transferred, result]() mutable { transferred.MarkFinished(result); });
      if (!spawn_status.ok()) {
        transferred.MarkFinished(spawn_status);
      }
    };
    auto callback_factory = [&callback]() { return callback; };
    if (future.TryAddCallback(callback_factory)) {
      return transferred;
    }
    // If the future is already finished and we aren't going to force spawn a thread
    // then we don't need to add another layer of callback and can return the original
    // future
    return future;
  }

  // Submit a callable and arguments for execution.  Return a future that
  // will return the callable's result value once.
  // The callable's arguments are copied before execution.
  template <typename Function, typename... Args,
            typename FutureType = typename ::arrow::detail::ContinueFuture::ForSignature<
                Function && (Args && ...)>>
  Result<FutureType> Submit(TaskHints hints, StopToken stop_token, Function&& func,
                            Args&&... args) {
    using ValueType = typename FutureType::ValueType;

    auto future = FutureType::Make();
    auto task = std::bind(::arrow::detail::ContinueFuture{}, future,
                          std::forward<Function>(func), std::forward<Args>(args)...);
    struct {
      WeakFuture<ValueType> weak_fut;

      void operator()(const Status& st) {
        auto fut = weak_fut.get();
        if (fut.is_valid()) {
          fut.MarkFinished(st);
        }
      }
    } stop_callback{WeakFuture<ValueType>(future)};
    ARROW_RETURN_NOT_OK(SpawnReal(hints, std::move(task), std::move(stop_token),
                                  std::move(stop_callback)));

    return future;
  }

  template <typename Function, typename... Args,
            typename FutureType = typename ::arrow::detail::ContinueFuture::ForSignature<
                Function && (Args && ...)>>
  Result<FutureType> Submit(StopToken stop_token, Function&& func, Args&&... args) {
    return Submit(TaskHints{}, stop_token, std::forward<Function>(func),
                  std::forward<Args>(args)...);
  }

  template <typename Function, typename... Args,
            typename FutureType = typename ::arrow::detail::ContinueFuture::ForSignature<
                Function && (Args && ...)>>
  Result<FutureType> Submit(TaskHints hints, Function&& func, Args&&... args) {
    return Submit(std::move(hints), StopToken::Unstoppable(),
                  std::forward<Function>(func), std::forward<Args>(args)...);
  }

  template <typename Function, typename... Args,
            typename FutureType = typename ::arrow::detail::ContinueFuture::ForSignature<
                Function && (Args && ...)>>
  Result<FutureType> Submit(Function&& func, Args&&... args) {
    return Submit(TaskHints{}, StopToken::Unstoppable(), std::forward<Function>(func),
                  std::forward<Args>(args)...);
  }

  // Return the level of parallelism (the number of tasks that may be executed
  // concurrently).  This may be an approximate number.
  virtual int GetCapacity() = 0;

  // Returns whether the thread pool has a worker thread (or can spawn a worker thread) to
  // handle a task immediately.
  //
  // Returns falls if the thread pool is busy and a task submission would be queued
  //
  // Can be used by planners to determine if a task should be scheduled or run
  // synchronously
  //
  // This is a best-effort guess and could change between this call and actually
  // scheduling a task.
  virtual bool HasIdleCapacity() = 0;

 protected:
  ARROW_DISALLOW_COPY_AND_ASSIGN(Executor);

  Executor() = default;

  // Subclassing API
  virtual Status SpawnReal(TaskHints hints, FnOnce<void()> task, StopToken,
                           StopCallback&&) = 0;
};

}  // namespace internal
}  // namespace arrow