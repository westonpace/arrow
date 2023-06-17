#include "arrow/compute/function_options.h"

#include "arrow/buffer.h"
#include "arrow/compute/registry.h"
#include "arrow/compute/type_fwd.h"

namespace arrow {

namespace compute {

Result<std::shared_ptr<Buffer>> FunctionOptionsType::Serialize(
    const FunctionOptions&) const {
  return Status::NotImplemented("Serialize for ", type_name());
}

Result<std::unique_ptr<FunctionOptions>> FunctionOptionsType::Deserialize(
    const Buffer& buffer) const {
  return Status::NotImplemented("Deserialize for ", type_name());
}

std::string FunctionOptions::ToString() const { return options_type()->Stringify(*this); }

bool FunctionOptions::Equals(const FunctionOptions& other) const {
  if (this == &other) return true;
  if (options_type() != other.options_type()) return false;
  return options_type()->Compare(*this, other);
}

std::unique_ptr<FunctionOptions> FunctionOptions::Copy() const {
  return options_type()->Copy(*this);
}

Result<std::shared_ptr<Buffer>> FunctionOptions::Serialize() const {
  return options_type()->Serialize(*this);
}

Result<std::unique_ptr<FunctionOptions>> FunctionOptions::Deserialize(
    const std::string& type_name, const Buffer& buffer) {
  ARROW_ASSIGN_OR_RAISE(auto options,
                        GetFunctionRegistry()->GetFunctionOptionsType(type_name));
  return options->Deserialize(buffer);
}

void PrintTo(const FunctionOptions& options, std::ostream* os) {
  *os << options.ToString();
}

}  // namespace compute
}  // namespace arrow
