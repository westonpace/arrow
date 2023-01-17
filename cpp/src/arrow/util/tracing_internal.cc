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

#include "arrow/util/tracing_internal.h"
#include "arrow/io/interfaces.h"
#include "arrow/util/thread_pool.h"
#include "arrow/util/tracing.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <thread>

#ifdef ARROW_WITH_OPENTELEMETRY

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4522)
#endif
#include <google/protobuf/util/json_util.h>

#include <opentelemetry/exporters/ostream/span_exporter.h>
#include <opentelemetry/exporters/otlp/otlp_http_exporter.h>
#include <opentelemetry/exporters/otlp/otlp_recordable_utils.h>
#include <opentelemetry/sdk/trace/batch_span_processor.h>
#include <opentelemetry/sdk/trace/recordable.h>
#include <opentelemetry/sdk/trace/span_data.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>
#include <opentelemetry/trace/provider.h>

#include <opentelemetry/exporters/otlp/protobuf_include_prefix.h>
#include <opentelemetry/exporters/otlp/protobuf_include_suffix.h>
#include <opentelemetry/proto/collector/trace/v1/trace_service.pb.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif

#include "arrow/util/io_util.h"
#include "arrow/util/logging.h"

namespace arrow {
namespace internal {
namespace tracing {

#ifdef ARROW_WITH_OPENTELEMETRY

namespace nostd = opentelemetry::nostd;
namespace otel = opentelemetry;

constexpr char kTracingBackendEnvVar[] = "ARROW_TRACING_BACKEND";

namespace {

namespace sdktrace = opentelemetry::sdk::trace;

// Custom JSON exporter. Leverages the OTLP HTTP exporter's utilities
// to log the same format that would be sent to OTLP.
class OtlpOStreamExporter final : public sdktrace::SpanExporter {
 public:
  explicit OtlpOStreamExporter(std::basic_ostream<char>* out) : out_(out) {
    protobuf_json_options_.add_whitespace = false;
  }

  std::unique_ptr<sdktrace::Recordable> MakeRecordable() noexcept override {
    // The header for the Recordable definition is not installed, work around that
    return exporter_.MakeRecordable();
  }
  otel::sdk::common::ExportResult Export(
      const nostd::span<std::unique_ptr<sdktrace::Recordable>>& spans) noexcept override {
    opentelemetry::proto::collector::trace::v1::ExportTraceServiceRequest request;
    otel::exporter::otlp::OtlpRecordableUtils::PopulateRequest(spans, &request);

    for (const auto& spans : request.resource_spans()) {
      std::string output;
      auto status = google::protobuf::util::MessageToJsonString(spans, &output,
                                                                protobuf_json_options_);
      if (ARROW_PREDICT_FALSE(!status.ok())) {
        return otel::sdk::common::ExportResult::kFailure;
      }
      (*out_) << output << std::endl;
    }

    return otel::sdk::common::ExportResult::kSuccess;
  }
  bool Shutdown(std::chrono::microseconds timeout =
                    std::chrono::microseconds(0)) noexcept override {
    return exporter_.Shutdown(timeout);
  }

 private:
  std::basic_ostream<char>* out_;
  opentelemetry::exporter::otlp::OtlpHttpExporter exporter_;
  google::protobuf::util::JsonPrintOptions protobuf_json_options_;
};

class ThreadIdSpanProcessor : public sdktrace::BatchSpanProcessor {
 public:
  using sdktrace::BatchSpanProcessor::BatchSpanProcessor;
  void OnEnd(std::unique_ptr<sdktrace::Recordable>&& span) noexcept override {
    std::stringstream thread_id;
    thread_id << std::this_thread::get_id();
    span->SetAttribute("thread.id", thread_id.str());
    sdktrace::BatchSpanProcessor::OnEnd(std::move(span));
  }
};

std::unique_ptr<sdktrace::SpanExporter> InitializeExporter() {
  auto maybe_env_var = arrow::internal::GetEnvVar(kTracingBackendEnvVar);
  if (maybe_env_var.ok()) {
    auto env_var = maybe_env_var.ValueOrDie();
    if (env_var == "ostream") {
      return std::make_unique<otel::exporter::trace::OStreamSpanExporter>();
    } else if (env_var == "otlp_http") {
      namespace otlp = opentelemetry::exporter::otlp;
      otlp::OtlpHttpExporterOptions opts;
      return std::make_unique<otlp::OtlpHttpExporter>(opts);
    } else if (env_var == "arrow_otlp_stdout") {
      return std::make_unique<OtlpOStreamExporter>(&std::cout);
    } else if (env_var == "arrow_otlp_stderr") {
      return std::make_unique<OtlpOStreamExporter>(&std::cerr);
    } else if (!env_var.empty()) {
      ARROW_LOG(WARNING) << "Requested unknown backend " << kTracingBackendEnvVar << "="
                         << env_var;
    }
  }
  return nullptr;
}

struct StorageSingleton : public Executor::Resource {
  StorageSingleton()
      : storage_(otel::context::RuntimeContext::GetConstRuntimeContextStorage()) {}
  nostd::shared_ptr<const otel::context::RuntimeContextStorage> storage_;
};

std::shared_ptr<Executor::Resource> GetStorageSingleton() {
  static std::shared_ptr<StorageSingleton> storage_singleton =
      std::make_shared<StorageSingleton>();
  return storage_singleton;
}

nostd::shared_ptr<sdktrace::TracerProvider> InitializeSdkTracerProvider() {
  // Bind the lifetime of the OT runtime context to the CPU and I/O thread
  // pools.  This will keep OT alive until all thread tasks have finished.
  internal::GetCpuThreadPool()->KeepAlive(GetStorageSingleton());
  io::default_io_context().executor()->KeepAlive(GetStorageSingleton());
  auto exporter = InitializeExporter();
  if (exporter) {
    sdktrace::BatchSpanProcessorOptions options;
    options.max_queue_size = 16384;
    options.schedule_delay_millis = std::chrono::milliseconds(500);
    options.max_export_batch_size = 16384;
    auto processor =
        std::make_unique<ThreadIdSpanProcessor>(std::move(exporter), options);
    return std::make_shared<sdktrace::TracerProvider>(std::move(processor));
  }
  return nostd::shared_ptr<sdktrace::TracerProvider>();
}

class FlushLog {
 public:
  explicit FlushLog(nostd::shared_ptr<sdktrace::TracerProvider> provider)
      : provider_(std::move(provider)) {}
  ~FlushLog() {
    // TODO: ForceFlush apparently sends data that OTLP connector can't handle
    // if (provider_) {
    //   provider_->ForceFlush(std::chrono::microseconds(1000000));
    // }
  }
  nostd::shared_ptr<sdktrace::TracerProvider> provider_;
};

nostd::shared_ptr<sdktrace::TracerProvider> GetSdkTracerProvider() {
  static FlushLog flush_log = FlushLog(InitializeSdkTracerProvider());
  return flush_log.provider_;
}

nostd::shared_ptr<otel::trace::TracerProvider> InitializeTracing() {
  nostd::shared_ptr<otel::trace::TracerProvider> provider = GetSdkTracerProvider();
  if (provider) otel::trace::Provider::SetTracerProvider(provider);
  return otel::trace::Provider::GetTracerProvider();
}

otel::trace::TracerProvider* GetTracerProvider() {
  static nostd::shared_ptr<otel::trace::TracerProvider> provider = InitializeTracing();
  return provider.get();
}
}  // namespace

opentelemetry::trace::Tracer* GetTracer() {
  static nostd::shared_ptr<opentelemetry::trace::Tracer> tracer =
      GetTracerProvider()->GetTracer("arrow");
  return tracer.get();
}

opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& UnwrapSpan(
    ::arrow::util::tracing::SpanDetails* span) {
  SpanImpl* span_impl = checked_cast<SpanImpl*>(span);
  ARROW_CHECK(span_impl->ot_span)
      << "Attempted to dereference a null pointer. Use Span::Set before "
         "dereferencing.";
  return span_impl->ot_span;
}

const opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& UnwrapSpan(
    const ::arrow::util::tracing::SpanDetails* span) {
  const SpanImpl* span_impl = checked_cast<const SpanImpl*>(span);
  ARROW_CHECK(span_impl->ot_span)
      << "Attempted to dereference a null pointer. Use Span::Set before "
         "dereferencing.";
  return span_impl->ot_span;
}

opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& RewrapSpan(
    ::arrow::util::tracing::SpanDetails* span,
    opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> ot_span) {
  SpanImpl* span_impl = checked_cast<SpanImpl*>(span);
  span_impl->ot_span = std::move(ot_span);
  return span_impl->ot_span;
}

opentelemetry::trace::StartSpanOptions SpanOptionsWithParent(
    const util::tracing::Span& parent_span) {
  opentelemetry::trace::StartSpanOptions options;
  options.parent = UnwrapSpan(parent_span.details.get())->GetContext();
  return options;
}

#elif defined(ARROW_WITH_DEBUG_TRACING)

constexpr int64_t kNoSpanId = -1;

constexpr char kTracingEnabledVar[] = "ARROW_TRACING";

bool DetermineIfTracingRequested() {
  auto maybe_env_var = arrow::internal::GetEnvVar(kTracingEnabledVar);
  return maybe_env_var.ok();
}

bool tracing_enabled() {
  static bool tracing_enabled = DetermineIfTracingRequested();
  return tracing_enabled;
}

int64_t SpanId() {
  static std::atomic<int64_t> id_counter(0);
  return id_counter++;
}

SpanImpl::SpanImpl(std::string_view name)
    : id(SpanId()), start(std::chrono::high_resolution_clock::now()), name(name) {}

SpanImpl::~SpanImpl() {
  if (!ended) {
    RecordSpanEnd(*this);
  }
}

// Important during move to mark the other end as ended so it does not log itself
SpanImpl::SpanImpl(SpanImpl&& other) {
  id = other.id;
  name = std::move(other.name);
  start = other.start;
  ended = other.ended;
  other.ended = true;
}

SpanImpl& SpanImpl::operator=(SpanImpl&& other) {
  id = other.id;
  name = std::move(other.name);
  start = other.start;
  ended = other.ended;
  other.ended = true;
  return *this;
}

void PrintTimeIso8601(std::chrono::high_resolution_clock::time_point time,
                      std::ostream& stream) {
  std::time_t as_time = std::chrono::system_clock::to_time_t(time);
  std::tm* as_tm = std::localtime(&as_time);
  long long as_ts =
      std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch())
          .count();

  stream << std::setfill('0') << std::put_time(as_tm, "%FT%H:%M:") << std::setw(2)
         << (as_ts / 1000) % 60 << '.' << std::setw(3) << as_ts % 1000;
}

std::stringstream NewLogLine(int64_t span_id,
                             std::chrono::high_resolution_clock::time_point now =
                                 std::chrono::high_resolution_clock::now()) {
  std::stringstream statement_builder;
  PrintTimeIso8601(now, statement_builder);
  statement_builder << " " << std::hex << std::this_thread::get_id() << std::dec;
  statement_builder << " " << span_id;
  return statement_builder;
}

void TraceWrite(std::string message) {
  if (tracing_enabled()) {
    std::cout << message;
  }
}

void RecordSpanStart(util::tracing::Span& span, std::string_view name) {
  auto span_impl = std::make_unique<SpanImpl>(name);

  auto line = NewLogLine(span_impl->id, span_impl->start);
  span.details = std::move(span_impl);
  line << " StartSpan(" << name << ")" << std::endl;
  TraceWrite(line.str());
}

Scope ScopedRecordSpanStart(util::tracing::Span& span, std::string_view name) {
  RecordSpanStart(span, name);
  return Scope{std::move(span.details)};
}

void RecordSpanEnd(SpanImpl& span_impl) {
  if (span_impl.ended) {
    return;
  }
  span_impl.ended = true;
  auto now = std::chrono::high_resolution_clock::now();
  auto duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - span_impl.start).count();
  auto line = NewLogLine(span_impl.id, now);
  line << " FinishSpan(" << span_impl.name << ") duration=" << duration_ns << "ns"
       << std::endl;
  TraceWrite(line.str());
}

void RecordSpanEnd(util::tracing::Span& span) {
  SpanImpl& span_impl = checked_cast<SpanImpl&>(*span.details);
  RecordSpanEnd(span_impl);
}

void RecordSpanEvent(const util::tracing::Span& span, std::string_view event) {
  if (!span.valid()) {
    RecordSpanEvent(event);
    return;
  }
  SpanImpl& span_impl = checked_cast<SpanImpl&>(*span.details);
  auto line = NewLogLine(span_impl.id);
  line << " Event(" << span_impl.name << ") " << event << std::endl;
  TraceWrite(line.str());
}

void RecordSpanEvent(std::string_view event) {
  auto line = NewLogLine(kNoSpanId);
  line << " Event(?) " << event << std::endl;
  TraceWrite(line.str());
}

void RecordSpanError(const util::tracing::Span& span, const Status& st) {
  if (st.ok()) {
    return;
  }
  SpanImpl& span_impl = checked_cast<SpanImpl&>(*span.details);
  auto line = NewLogLine(span_impl.id);
  line << " Error(" << span_impl.name << ") " << st.ToString() << std::endl;
  TraceWrite(line.str());
}

#endif

}  // namespace tracing
}  // namespace internal
}  // namespace arrow
