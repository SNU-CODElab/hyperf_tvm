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

namespace {
class VarCollector : public StmtExprVisitor {
 public:
  explicit VarCollector() {}
  void VisitExpr_(const VarNode* op) final {
    for (auto v : vars) {
      if (op->name_hint == v) {
        return;
      }
    }
    vars.push_back(op->name_hint);
    StmtExprVisitor::VisitExpr_(op);
  }
  Array<String> vars;
};

class VarReplacer : public StmtExprMutator {
 public:
  explicit VarReplacer(Array<String> vars, const Buffer& bin_ptr, const Array<String>& skip_bufs)
      : vars_(vars), bin_ptr_(bin_ptr), skip_bufs_(skip_bufs) {}
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = Downcast<Var>(StmtExprMutator::VisitExpr_(op));
    for (auto v : vars_) {
      if (op->name_hint == v && !IsSkipVar(op->name_hint)) {
        return BufferLoad(bin_ptr_, Array<PrimExpr>{var});
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    if (IsSkipVar(op->buffer->name)) {
      return GetRef<PrimExpr>(op);
    }
    return StmtExprMutator::VisitExpr_(op);
  }
  bool IsSkipVar(String var) {
    for (auto v : skip_bufs_) {
      if (v == var) {
        return true;
      }
    }
    return false;
  }
  Buffer bin_ptr_;
  Array<String> vars_;
  Array<String> skip_bufs_;
};
}  // namespace

class BinningApplication : public StmtExprMutator {
 public:
  // Toplevel (static) function
  static Stmt Transform(const Stmt& stmt, String rowptr, String output, const Buffer& bin) {
    BinningApplication transformer(stmt, rowptr, output, bin);
    return transformer.VisitStmt(stmt);
  }

 protected:
  // Constructor
  BinningApplication(const Stmt& stmt, String rowptr, String output, const Buffer& bin)
      : initial_body_(stmt),
        rowptr_(rowptr),
        output_(output),
        bin_(bin),
        alloc_bufs_({bin->name}) {}

 private:
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    BufferLoad stmt = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    // rowptr
    if (op->buffer->name == rowptr_) {
      BufferLoadNode* n = stmt.CopyOnWrite();
      collector_(stmt);
      VarReplacer replacer(collector_.vars, bin_, alloc_bufs_);
      stmt = Downcast<BufferLoad>(replacer(stmt));
    }
    return std::move(stmt);
  }
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore stmt = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    // output
    if (op->buffer->name == output_) {
      VarReplacer replacer(collector_.vars, bin_, alloc_bufs_);
      stmt = Downcast<BufferStore>(replacer(stmt));
    }
    return std::move(stmt);
  }
  Stmt VisitStmt_(const BlockNode* op) final {
    for (auto buf : op->alloc_buffers) {
      alloc_bufs_.push_back(buf->name);
    }
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    for (int i = 0; i < op->alloc_buffers.size(); i++) {
      alloc_bufs_.pop_back();
    }
    return std::move(stmt);
  }
  Stmt VisitStmt_(const AllocateNode* op) final {
    alloc_bufs_.push_back(op->buffer_var->name_hint);
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const DeclBufferNode* op) final {
    alloc_bufs_.push_back(op->buffer->name);
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt initial_body_;
  String rowptr_;
  String output_;
  Buffer bin_;
  VarCollector collector_;
  Array<String> alloc_bufs_;
};

namespace transform {

Pass ApplyBinning(String rowptr, String output, String bin) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    for (unsigned i = 0; i < f->params.size(); i++) {
      Var p = n->params[i];
      const Buffer& buf = n->buffer_map[p];
      Var b = buf->data;
      if (b->name_hint == bin) {
        n->body = BinningApplication::Transform(std::move(f->body), rowptr, output, buf);
      }
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ApplyBinning", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ApplyBinning").set_body_typed(ApplyBinning);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
