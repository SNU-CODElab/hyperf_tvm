/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file convert_block_to_opaque.cc
 * \brief Convert the blocks to opaque blocks which do not have block vars.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ir/type.h>
#include <tvm/relay/expr.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target_info.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <map>
#include <unordered_map>
#include <unordered_set>

#include "ir_utils.h"

namespace tvm {
namespace tir {

PrimFunc ChangeBufferShape(PrimFunc f, int row) {
  auto* n = f.CopyOnWrite();
  // last param
  Var p = f->params.back();
  Optional<Array<Integer>> bcsr_parameters = f->GetAttr("bcsr_parameters", Array<Integer>({}));
  int R = bcsr_parameters.value()[0]->value;
  row = (row + R - 1) / R;
  tir::Buffer buf = n->buffer_map[p];
  tir::Var b = buf->data;
  auto writer = buf.CopyOnWrite();
  Array<PrimExpr> new_shape;
  new_shape.push_back(row);
  for (size_t i = 1; i < writer->shape.size(); i++) {
    new_shape.push_back(buf->shape[i]);
  }
  writer->shape = new_shape;
  n->buffer_map.Set(p, buf);
  return f;
}

namespace transform {

Pass ChangeBufferShape(int row) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return ChangeBufferShape(std::move(f), row);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ChangeBufferShape", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ChangeBufferShape").set_body_typed(ChangeBufferShape);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
