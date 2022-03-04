#ifndef SAFE_CALL_INTO_R_INCLUDED
#define SAFE_CALL_INTO_R_INCLUDED

#include <functional>
#include "./arrow_types.h"

#include <arrow/util/thread_pool.h>

arrow::internal::Executor*& GetRExecutor() {
  static arrow::internal::Executor* r_executor = nullptr;
  return r_executor;
}

/**
 * This is called at the top level, from R, when calling a function that might make
 * callbacks back into R.  For example:
 *
 * std::shared_ptr<arrow::Table> parquet___arrow___FileReader__ReadTable1(
 *  const std::shared_ptr<parquet::arrow::FileReader>& reader) {
 *   return RunWithCapturedR([&reader] {
 *     return reader->ReadTableAsync());
 *   });
 * }
 *
 * Note: ReadTableAsync does not exist today.  This would be the first case of exposing
 * C++ async functionality.  It wouldn't be too hard to create but you can maybe see
 * that this is opening a slight can of worms.
 *
 * Unfortunately, if we don't call the async version, we have no way in the C++ code to
 * setup an event loop.
 */
template <typename T>
arrow::Result<T> RunWithCapturedR(std::function<arrow::Future<T>()> task) {
  if (GetRExecutor() != nullptr) {
    // Error here that RunWithCapturedR called reentrantly which is a no-go
  }
  arrow::Result<T> cmd_result = arrow::internal::SerialExecutor::RunInSerialExecutor<T>(
      [task](arrow::internal::Executor* executor) {
        GetRExecutor() = executor;
        arrow::Future<T> result = task();
        return result;
      });
  GetRExecutor() = nullptr;
}

/**
 * This is called from an Arrow context when you need to call back into R.  For example,
 * you would call this from your filesystem adapter when you need to make R calls to get
 * the data.
 *
 * Note that this function returns a Future<T>.  That's because we are going to
 * "pause/yield" this Arrow thread while we wait for the R thread to do its thing.
 *
 * Fortunately, the filesystem API already has support for Future-returning API methods.
 * You can then simulate the sync version by returning (e.g.) ReadAtAsync().result();
 */
template <typename T>
arrow::Future<T> SafeCallIntoR(std::function<T()> task) {
  if (GetRExecutor() == nullptr) {
    // Error here that we are not in a RunWithCapturedR context
  }
  return arrow::DeferNotOk((GetRExecutor())->Submit(task));
}

#endif
