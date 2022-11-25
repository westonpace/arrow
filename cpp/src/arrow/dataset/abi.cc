#include "arrow/dataset/abi.h"

#include "arrow/dataset/plan.h"

extern "C" void arrow_dataset_initialize() { arrow::dataset::internal::Initialize(); }
