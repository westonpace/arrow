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
#include <iostream>

#include "arrow/engine/substrait/options.h"

#include <google/protobuf/util/json_util.h>
#include "arrow/compute/exec/options.h"
#include "arrow/engine/substrait/relation_internal.h"
#include "substrait/extension_rels.pb.h"

namespace arrow {
namespace engine {

class DefaultExtensionProvider : public ExtensionProvider {
 public:
  Result<DeclarationInfo> HandleMulti(const std::vector<DeclarationInfo>& inputs,
                                      const google::protobuf::Any& rel) override {
    return Status::NotImplemented("No support for multi-extensions in Substrait");
  }
  Result<DeclarationInfo> HandleSingle(const DeclarationInfo& input,
                                       const google::protobuf::Any& rel) override {
    if (rel.Is<arrow::substrait::DelayRel>()) {
      arrow::substrait::DelayRel delay_rel;
      rel.UnpackTo(&delay_rel);
      // Here is where you would create an as-of join rel but instead we will just create
      // a pretend project rel that adds a new field
      std::vector<compute::Expression> exprs;
      std::vector<std::string> names;

      std::vector<std::shared_ptr<Field>> fields = input.output_schema->fields();
      for (const auto& field : fields) {
        exprs.push_back(compute::field_ref(field->name()));
        names.push_back(field->name());
      }
      exprs.push_back(compute::literal(delay_rel.seconds()));
      names.push_back("delay");

      fields.push_back(field("delay", float64()));

      compute::ProjectNodeOptions project_node_opts{std::move(exprs), std::move(names)};
      return DeclarationInfo{
          compute::Declaration::Sequence(
              {input.declaration, {"project", std::move(project_node_opts)}}),
          schema(std::move(fields))};
    }
    return Status::NotImplemented("Unrecognized single-extension in Susbstrait plan: ",
                                  rel.DebugString());
  }
  Result<DeclarationInfo> HandleLeaf(const google::protobuf::Any& rel) override {
    return Status::NotImplemented("No support for leaf-extensions in Substrait");
  }
};

std::shared_ptr<ExtensionProvider> ExtensionProvider::kDefaultExtensionProvider =
    std::make_shared<DefaultExtensionProvider>();

}  // namespace engine
}  // namespace arrow