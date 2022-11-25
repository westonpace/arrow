#include "arrow/engine/abi.h"

#include <cerrno>
#include <iostream>

#include "arrow/buffer.h"
#include "arrow/c/bridge.h"
#include "arrow/compute/exec/exec_plan.h"
#include "arrow/engine/substrait/serde.h"
#include "arrow/record_batch.h"
#include "arrow/result.h"
#include "arrow/util/macros.h"

arrow::Result<ArrowArrayStream> DoRun(void* plan, int64_t plan_length) {
  arrow::Buffer buffer(reinterpret_cast<uint8_t*>(plan), plan_length);
  ARROW_ASSIGN_OR_RAISE(arrow::engine::DeclarationInfo decl_info,
                        arrow::engine::DeserializePlan(buffer));
  ARROW_ASSIGN_OR_RAISE(std::unique_ptr<arrow::RecordBatchReader> reader,
                        arrow::compute::DeclarationToReader(decl_info.declaration));
  ArrowArrayStream out;
  arrow::Status export_status = arrow::ExportRecordBatchReader(std::move(reader), &out);
  if (export_status.ok()) {
    return out;
  } else {
    return export_status;
  }
}

int ErrnoFromStatus(const arrow::Status& st) {
  // TODO: Improve
  return EINVAL;
}

struct ErrorStreamPrivateData {
  int st_as_errno;
  std::string error_message;
};

ArrowArrayStream WrapError(const arrow::Status& st) {
  ArrowArrayStream wrapped;
  wrapped.private_data = new ErrorStreamPrivateData{ErrnoFromStatus(st), st.ToString()};
  wrapped.get_schema = [](ArrowArrayStream* stream, ArrowSchema*) {
    return reinterpret_cast<ErrorStreamPrivateData*>(stream->private_data)->st_as_errno;
  };
  wrapped.get_next = [](ArrowArrayStream* stream, ArrowArray*) {
    return reinterpret_cast<ErrorStreamPrivateData*>(stream->private_data)->st_as_errno;
  };
  wrapped.get_last_error = [](ArrowArrayStream* stream) {
    return reinterpret_cast<ErrorStreamPrivateData*>(stream->private_data)
        ->error_message.c_str();
  };
  wrapped.release = [](ArrowArrayStream* stream) {
    auto* private_data = reinterpret_cast<ErrorStreamPrivateData*>(stream->private_data);
    delete private_data;
  };
  return wrapped;
}

ArrowArrayStream arrow_substrait_run(void* plan, int64_t plan_length,
                                     ArrowArrayStream* named_tables,
                                     int num_named_tables) {
  std::cout << "plan=" << plan << std::endl;
  std::cout << "plan_length=" << plan_length << std::endl;
  std::cout << "named_tables=" << named_tables << std::endl;
  std::cout << "num_named_tables=" << num_named_tables << std::endl;
  auto* bytes = reinterpret_cast<uint8_t*>(plan);
  for (int64_t i = 0; i < plan_length; i++) {
    std::cout << " plan byte[" << i << "] = " << bytes[i] << std::endl;
  }
  arrow::Result<ArrowArrayStream> maybe_output_stream = DoRun(plan, plan_length);
  if (maybe_output_stream.ok()) {
    return *maybe_output_stream;
  } else {
    return WrapError(maybe_output_stream.status());
  }
}
