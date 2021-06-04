#include <benchmark/benchmark.h>
#include <vector>

#include "arrow/result.h"
#include "arrow/status.h"

namespace arrow {

class PapiCounters {
 public:
  ~PapiCounters();

  Status Start();
  Status Pause();
  Status Resume();
  Status Finish();
  Status AddToBenchmark(benchmark::State& state);

  static Result<PapiCounters> Make(std::vector<std::string> counter_names);

  struct PapiState;

 private:
  explicit PapiCounters(std::vector<std::string> counter_names);
  Status Initialize();
  bool IsRunning();
  Status EnsureRunning();

  std::vector<std::string> counter_names_;
  std::shared_ptr<PapiState> papi_state_;
};

}  // namespace arrow