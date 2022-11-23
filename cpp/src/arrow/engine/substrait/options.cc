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
#include "arrow/compute/exec/asof_join_node.h"
#include "arrow/compute/exec/options.h"
#include "arrow/engine/substrait/expression_internal.h"
#include "arrow/engine/substrait/relation_internal.h"
#include "arrow/util/logging.h"
#include "substrait/extension_rels.pb.h"

namespace arrow {
namespace engine {

class DefaultExtensionProvider : public ExtensionProvider {
 public:
  Result<DeclarationInfo> MakeRel(const std::vector<DeclarationInfo>& inputs,
                                  const google::protobuf::Any& rel, bool* has_emit,
                                  std::vector<int>* emit_info,
                                  const ExtensionSet& ext_set) override {
    if (rel.Is<substrait_ext::AsOfJoinRel>()) {
      substrait_ext::AsOfJoinRel as_of_join_rel;
      rel.UnpackTo(&as_of_join_rel);
      return MakeAsOfJoinRel(inputs, as_of_join_rel, has_emit, emit_info, ext_set);
    }
    return Status::NotImplemented("Unrecognized extension in Susbstrait plan: ",
                                  rel.DebugString());
  }

 private:
  Result<DeclarationInfo> MakeAsOfJoinRel(
      const std::vector<DeclarationInfo>& inputs,
      const substrait_ext::AsOfJoinRel& as_of_join_rel, bool* has_emit,
      std::vector<int>* emit_info, const ExtensionSet& ext_set) {
    if (inputs.size() < 2) {
      return Status::Invalid("substrait::AsOfJoinNode too few input tables: ",
                             inputs.size());
    }
    if (static_cast<size_t>(as_of_join_rel.keys_size()) != inputs.size()) {
      return Status::Invalid("substrait::AsOfJoinNode mismatched number of inputs");
    }

    size_t n_input = inputs.size(), i = 0;
    std::vector<compute::AsofJoinNodeOptions::Keys> input_keys(n_input);
    std::unordered_map<int, int> canonical_indices_map;
    int left_side_on_idx = -1;
    int inputs_left_of_this = 0;
    std::vector<int> left_side_by_indices;
    for (const auto& keys : as_of_join_rel.keys()) {
      // on-key
      if (!keys.has_on()) {
        return Status::Invalid("substrait::AsOfJoinNode missing on-key for input ", i);
      }
      ARROW_ASSIGN_OR_RAISE(auto on_key_expr, FromProto(keys.on(), ext_set, {}));
      if (on_key_expr.field_ref() == NULLPTR) {
        return Status::NotImplemented(
            "substrait::AsOfJoinNode non-field-ref on-key for input ", i);
      }
      const FieldRef& on_key = *on_key_expr.field_ref();
      int on_idx = on_key.field_path()->indices()[0];
      if (i == 0) {
        left_side_on_idx = on_idx;
      } else {
        canonical_indices_map[inputs_left_of_this + on_idx] = left_side_on_idx;
      }

      // by-key
      std::vector<FieldRef> by_key;
      for (int key_idx = 0; key_idx < keys.by_size(); key_idx++) {
        const auto& by_item = keys.by(key_idx);
        ARROW_ASSIGN_OR_RAISE(auto by_key_expr, FromProto(by_item, ext_set, {}));
        if (by_key_expr.field_ref() == NULLPTR) {
          return Status::NotImplemented(
              "substrait::AsOfJoinNode non-field-ref by-key for input ", i);
        }
        by_key.push_back(*by_key_expr.field_ref());
        int by_idx = by_key[by_key.size() - 1].field_path()->indices()[0];
        if (i == 0) {
          left_side_by_indices.push_back(by_idx);
        } else {
          canonical_indices_map[inputs_left_of_this + by_idx] =
              left_side_by_indices[key_idx];
        }
      }

      input_keys[i] = {std::move(on_key), std::move(by_key)};
      inputs_left_of_this += inputs[i].output_schema->num_fields();
      ++i;
    }

    // schema
    int64_t tolerance = as_of_join_rel.tolerance();
    std::vector<std::shared_ptr<Schema>> input_schema(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      input_schema[i] = inputs[i].output_schema;
    }
    ARROW_ASSIGN_OR_RAISE(auto schema,
                          compute::asofjoin::MakeOutputSchema(input_schema, input_keys));
    compute::AsofJoinNodeOptions asofjoin_node_opts{std::move(input_keys), tolerance};

    // declaration
    std::vector<compute::Declaration::Input> input_decls(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      input_decls[i] = inputs[i].declaration;
    }

    // emit info
    // substrait expects asof join to repeat the on/by keys for each table.  The node
    // does not do this.  Here the emit info is patched to duplicate the on/by keys
    // from the first table

    std::vector<int> skipped;
    for (const auto& canonical_index : canonical_indices_map) {
      skipped.push_back(canonical_index.first);
    }
    std::sort(skipped.begin(), skipped.end());
    std::size_t total_num_fields = skipped.size() + schema->fields().size();
    std::vector<int> substrait_idx_to_asof_idx(total_num_fields);
    int skipped_idx = 0;
    for (std::size_t idx = 0; idx < substrait_idx_to_asof_idx.size(); idx++) {
      if (skipped[skipped_idx] == static_cast<int>(idx)) {
        skipped_idx++;
        substrait_idx_to_asof_idx[idx] =
            canonical_indices_map.find(static_cast<int>(idx))->second;
      } else {
        substrait_idx_to_asof_idx[idx] = static_cast<int>(idx) - skipped_idx;
      }
    }

    if (*has_emit) {
      for (std::size_t k = 0; k < emit_info->size(); k++) {
        (*emit_info)[k] = substrait_idx_to_asof_idx[(*emit_info)[k]];
      }
    } else {
      emit_info->reserve(substrait_idx_to_asof_idx.size());
      *has_emit = true;
      for (int idx : substrait_idx_to_asof_idx) {
        emit_info->push_back(idx);
      }
    }

    return DeclarationInfo{
        compute::Declaration("asofjoin", input_decls, std::move(asofjoin_node_opts)),
        schema};
  }
};

std::shared_ptr<ExtensionProvider> ExtensionProvider::kDefaultExtensionProvider =
    std::make_shared<DefaultExtensionProvider>();

}  // namespace engine
}  // namespace arrow
