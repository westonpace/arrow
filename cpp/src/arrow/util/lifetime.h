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

#include <memory>

#include "arrow/util/functional.h"

/// User-defined extension types.
namespace arrow {

/// @brief Shuts down the Arrow library.  This method should be called
///        when you are finished using Arrow.
///
///        This method will halt all background Arrow tasks.  In addition
///        Arrow will shutdown any 3rd party libraries (e.g. S3) which require
///        shutdown.
///
///        Failure to call this could lead to crashes on exit caused by Arrow
///        background tasks attempting to use resources that are being destroyed.
void Shutdown();

namespace util {
namespace internal {

/// An enum to prioritize destruction of resources during the call to arrow::Shutdown
enum class ShutdownPriority {
  kThreadingResource = 0,  // Shut down first
  kPostThreading           // Only shutdown after all threading resources are shut down
};

/// Register a task to be called at arrow::Shutdown
void AddShutdownTask(::arrow::internal::FnOnce<void()> destroy,
                     ShutdownPriority priority = ShutdownPriority::kPostThreading);

/// Sanity check to make sure we are not shutdown.  Will assert in debug mode if shutdown
/// has been called.
void CheckNotShutdown();

/// @brief helper class to wrap any objects which should be destroyed
///        at shutdown
///
/// Generally this should not be needed.  Objects with static
/// duration will be torn down normally at exit.  Only use this
/// for objects which need to be destroyed earlier for some reason.
///
/// Example usage:
///
/// FooService* GetGlobalFooService() {
///   // This FooService instance will be destroyed on a call to arrow::Shutdown
///   // after all thread tasks have finished.  If a user doesn't all arrow::Shutdown
///   // it will be destroyed during the normal shutdown process in an undetermined order.
///   static internal::ArrowSingleton<FooService> foo_service(CreateFooService());
///   return foo_service.Get();
/// }
template <typename T>
class ArrowSingleton {
 public:
  ArrowSingleton(std::unique_ptr<T> resource,
                 ShutdownPriority priority = ShutdownPriority::kPostThreading)
      : resource_(std::move(resource)) {
    AddShutdownTask([this] { resource_.reset(); });
  }

  ArrowSingleton(std::shared_ptr<T> resource,
                 ShutdownPriority = ShutdownPriority::kPostThreading)
      : resource_(std::move(resource)) {
    AddShutdownTask([this] { resource_.reset(); });
  }

  T* Get() {
    CheckNotShutdown();
    return resource_.get();
  }

 private:
  std::shared_ptr<T> resource_;
};

}  // namespace internal
}  // namespace util
}  // namespace arrow
