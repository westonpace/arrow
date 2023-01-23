#include "arrow/util/lifetime.h"

#include <mutex>
#include <vector>

#include "arrow/util/logging.h"

namespace arrow {

namespace util {
namespace internal {

namespace {
class ShutdownContainer {
 public:
  void Reset() {
    std::call_once(shutdown_flag, [this] {
      is_shutdown = true;
      for (auto& resource : threading_resources) {
        std::move(resource)();
      }
      threading_resources.clear();
      for (auto& resource : post_threading_resources) {
        std::move(resource)();
      }
      post_threading_resources.clear();
    });
  }

  std::vector<::arrow::internal::FnOnce<void()>> threading_resources;
  std::vector<::arrow::internal::FnOnce<void()>> post_threading_resources;
  bool is_shutdown = false;
  std::once_flag shutdown_flag;
};

ShutdownContainer* GetShutdownContainer() {
  static ShutdownContainer shutdown_container;
  return &shutdown_container;
}

}  // namespace

void CheckNotShutdown() {
  DCHECK(!GetShutdownContainer()->is_shutdown) << "attempt to use arrow after shutdown";
}

void AddShutdownTask(::arrow::internal::FnOnce<void()> destroy,
                     ShutdownPriority priority) {
  ShutdownContainer* shutdown_container = GetShutdownContainer();
  DCHECK(!shutdown_container->is_shutdown)
      << "attempt to register singleton after calling shutdown (shutdown during "
         "initialization?)";
  switch (priority) {
    case ShutdownPriority::kThreadingResource:
      shutdown_container->threading_resources.push_back(std::move(destroy));
      break;
    case ShutdownPriority::kPostThreading:
      shutdown_container->post_threading_resources.push_back(std::move(destroy));
      break;
    default:
      DCHECK(false) << "attempt to register singleton with unknown priority";
  }
}

}  // namespace internal

void Shutdown() { internal::GetShutdownContainer()->Reset(); }

}  // namespace util
}  // namespace arrow
