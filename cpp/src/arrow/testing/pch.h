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

// Often-used headers, for precompiling.
// If updating this header, please make sure you check compilation speed
// before checking in.  Adding headers which are not used extremely often
// may incur a slowdown, since it makes the precompiled header heavier to load.

#include "arrow/pch.h"

#include "arrow/array/builder_adaptive.h"
#include "arrow/array/builder_base.h"
#include "arrow/array/builder_binary.h"
#include "arrow/array/builder_decimal.h"
#include "arrow/array/builder_dict.h"
#include "arrow/array/builder_nested.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/array/builder_run_end.h"
#include "arrow/array/builder_time.h"
#include "arrow/array/builder_union.h"
#include "arrow/builder.h"

#include "arrow/testing/gtest_util.h"
#include "arrow/testing/random.h"
#include "arrow/testing/util.h"

#include "arrow/compute/api_aggregate.h"
#include "arrow/compute/api_scalar.h"
#include "arrow/compute/api_vector.h"
