#include "arrow/util/papi_util.h"
#include "arrow/status.h"
#include "arrow/testing/gtest_util.h"

#include </usr/include/papi.h>

namespace arrow {

namespace {

Status PapiErrToStatus(int retval, int expected = PAPI_OK,
                       const std::string& message = "") {
  if (retval != expected) {
    if (message.empty()) {
      char* err = PAPI_strerror(retval);
      if (err == nullptr) {
        return Status::Invalid("Received unknown PAPI error code " +
                               std::to_string(retval));
      }
      return Status::Invalid(err);
    }
    return Status::Invalid(message);
  }
  return Status::OK();
}

bool gInitialized = false;

Status PapiInitialize() {
  return PapiErrToStatus(PAPI_library_init(PAPI_VER_CURRENT), PAPI_VER_CURRENT,
                         "PAPI initialization error");
}

Status EnsureInitialized() {
  if (!gInitialized) {
    gInitialized = true;
    return PapiInitialize();
  }
  return Status::OK();
}

}  // namespace

struct PapiCounters::PapiState {
  ~PapiState() {
    if (event_set != PAPI_NULL) {
      ABORT_NOT_OK(PapiErrToStatus(PAPI_cleanup_eventset(event_set)));
      ABORT_NOT_OK(PapiErrToStatus(PAPI_destroy_eventset(&event_set)));
      event_set = PAPI_NULL;
    }
  }

  int event_set = PAPI_NULL;
  std::vector<long long> values;
  std::vector<int> counter_to_event_idx;
};

PapiCounters::PapiCounters(std::vector<std::string> counter_names)
    : counter_names_(std::move(counter_names)),
      papi_state_(std::make_shared<PapiState>()) {}

PapiCounters::~PapiCounters() = default;

bool PapiCounters::IsRunning() {
  int status = 0;
  PAPI_state(papi_state_->event_set, &status);
  return (status & PAPI_RUNNING) != 0;
}

Status PapiCounters::EnsureRunning() {
  if (!IsRunning()) {
    return Status::Invalid(
        "PapiCounters::Start() must be called before calling this function");
  }
  return Status::OK();
}

Status PapiCounters::Initialize() {
  RETURN_NOT_OK(PapiErrToStatus(PAPI_create_eventset(&papi_state_->event_set)));
  for (const auto& counter_name : counter_names_) {
    int papi_code;
    int retval = PAPI_event_name_to_code(&*counter_name.begin(), &papi_code);
    if (retval == PAPI_OK) {
      papi_state_->counter_to_event_idx.push_back(papi_state_->values.size());
      papi_state_->values.push_back(0);
      RETURN_NOT_OK(PapiErrToStatus(PAPI_add_event(papi_state_->event_set, papi_code)));
    } else {
      papi_state_->counter_to_event_idx.push_back(-1);
    }
  }
  return Status::OK();
}

Result<PapiCounters> PapiCounters::Make(std::vector<std::string> counter_names) {
  RETURN_NOT_OK(EnsureInitialized());
  PapiCounters counters(std::move(counter_names));
  RETURN_NOT_OK(counters.Initialize());
  return counters;
}

Status PapiCounters::Start() {
  if (IsRunning()) {
    return Status::Invalid("PapiCounters::Start() called twice");
  }
  return PapiErrToStatus(PAPI_start(papi_state_->event_set));
}

Status PapiCounters::Pause() {
  RETURN_NOT_OK(EnsureRunning());
  return PapiErrToStatus(
      PAPI_accum(papi_state_->event_set, &*papi_state_->values.begin()));
}

Status PapiCounters::Resume() {
  RETURN_NOT_OK(EnsureRunning());
  return PapiErrToStatus(PAPI_reset(papi_state_->event_set));
}

Status PapiCounters::Finish() {
  // Pause first to capture counter values via accum
  RETURN_NOT_OK(Pause());
  long long unused[papi_state_->values.size()];
  return PapiErrToStatus(PAPI_stop(papi_state_->event_set, unused));
}

Status PapiCounters::AddToBenchmark(benchmark::State& state) {
  for (std::size_t i = 0; i < counter_names_.size(); i++) {
    const auto& counter_name = counter_names_[i];
    if (papi_state_->counter_to_event_idx[i] < 0) {
      state.counters[counter_name] = -1;
    } else {
      auto counter_idx = static_cast<std::size_t>(papi_state_->counter_to_event_idx[i]);
      state.counters[counter_name] = papi_state_->values[counter_idx];
    }
  }
  return Status::OK();
}

}  // namespace arrow