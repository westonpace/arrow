#include "safe-call-into-r.h"

#include <thread>

// [[arrow::export]]
cpp11::strings TestSafeCallIntoR() {
  // This simulates the Arrow thread pool.  Just imagine it is static and lives forever.
  std::thread* thread_ptr;

  // Pretend we are in parquet___arrow___FileReader__ReadTable1, you wouldn't need to
  // capture thread_ptr in the real world.
  arrow::Result<int> run_res = RunWithCapturedR<int>([&thread_ptr]() {
    // We wouldn't Make a future here, we would call ReadTableAsync and
    // return that future
    arrow::Future<int> fut = arrow::Future<int>::Make();
    thread_ptr = new std::thread([fut]() mutable {
      // At this point we are deep in the bowels of parquet reading and need to issue a
      // call to the filesystem.  So we are in your filesystem adapter here needing to
      // call into R
      SafeCallIntoR<int>([] {
        // The body of this task runs on the R thread, you can do R thread stuff here.
        cpp11::function gc = cpp11::package("base")["gc"];
        gc();
        return 0;
        // In reality you wouldn't have to worry about the AddCallback because you would
        // be implementing something like ReadAtAsync and you could just return a future.
      }).AddCallback([fut](const arrow::Result<int>&) mutable { fut.MarkFinished(0); });
    });
    return fut;
  });
  // Ignore everything below this point
  thread_ptr->join();
  delete thread_ptr;

  cpp11::writable::strings results_sexp;
  return results_sexp;
}
