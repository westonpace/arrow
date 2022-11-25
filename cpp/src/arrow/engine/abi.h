#pragma once

#include <stdint.h>

#include "arrow/c/abi.h"
#include "arrow/engine/substrait/visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ARROW_ENGINE_C_DATA_INTERFACE
#define ARROW_ENGINE_C_DATA_INTERFACE

ARROW_ENGINE_EXPORT ArrowArrayStream arrow_substrait_run(void* plan, int64_t plan_length,
                                                         ArrowArrayStream* named_tables,
                                                         int num_named_tables);

#endif

#ifdef __cplusplus
}
#endif
