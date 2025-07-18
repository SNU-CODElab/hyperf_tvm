/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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
 * Lower block init stmt into branch stmt
 * \file lower_reduction.cc
 */
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <iostream>
#include <set>

#include "../schedule/utils.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

namespace {
/*!
 * \brief Collect buffer information.
 */
class BufferCollector : public StmtExprVisitor {
 public:
  /*! \brief Map the buffer var to all aliased buffers. */
  Map<Var, Buffer> var2buffer_;
  ;

 private:
  void VisitExpr_(const BufferLoadNode* op) final {
    if (excluded_buffers_.count(op->buffer) == 0) {
      var2buffer_.Set(op->buffer->data, op->buffer);
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BlockNode* op) final {
    for (auto match_buffer : op->match_buffers) {
      excluded_buffers_.insert(match_buffer->buffer);
    }
    this->VisitStmt(op->body);
  }

  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> excluded_buffers_;
};
}  // namespace

// [ywshin]: 완벽하지 않은 구현이다. 우선 때워둔다.
class MatchBufferInliner : public StmtExprMutator {
 public:
  explicit MatchBufferInliner() {}

 private:
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    if (op->block->name_hint == "match_buffer") {
      return this->VisitStmt(op->block);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = GetRef<Block>(op);
    BlockNode* block_ptr = block.CopyOnWrite();
    for (auto match_buffer : block->match_buffers) {
      BufferRegion source = match_buffer->source;
      Array<PrimExpr> indices;
      for (auto range : source->region) {
        if (range->min.as<Var>().defined() && range->extent.as<IntImm>().defined() &&
            range->extent.as<IntImm>().value()->value == 1) {
          indices.push_back(Downcast<Var>(range->min));
        } else if (range->min.as<IntImm>() && range->min.as<IntImm>().value()->value == 0 &&
                   range->extent.as<IntImm>()) {
          indices.push_back(Integer(-1));
        } else {
          LOG(FATAL) << "T.match_buffer in the block is not well-formed";
          throw;
        }
      }
      subst_map_.Set(match_buffer->buffer->data, BufferLoad(source->buffer, indices));
    }
    if (block->name_hint == "match_buffer") {
      return this->VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    if (subst_map_.count(op->buffer->data) != 0) {
      BufferLoad n = subst_map_[op->buffer->data];
      Array<PrimExpr> indices;
      for (auto index : n->indices) {
        if (index.as<IntImm>().defined() && index.as<IntImm>().value()->value == -1) {
          if (op->indices.size() > 1) {
            LOG(FATAL) << "Too many indices";
            throw;
          }
          indices.push_back(op->indices[0]);
        } else {
          indices.push_back(index);
        }
      }
      n.CopyOnWrite()->indices = indices;
      return std::move(n);
    }
    return StmtExprMutator::VisitExpr_(op);
  }
  Map<Var, BufferLoad> subst_map_;
};

class PrivateBufferInliner : public StmtExprMutator {
 public:
  explicit PrivateBufferInliner(Array<String> buffer_names) {
    for (auto name : buffer_names) {
      buffer_names_.insert(name);
    }
  }

 private:
  Stmt VisitStmt_(const BufferStoreNode* op) final {
    String name = op->buffer->name;
    bool is_var_like_buffer =
        (op->buffer->shape.size() == 1 && op->buffer->shape[0].as<IntImm>().defined() &&
         op->buffer->shape[0].as<IntImm>().value()->value == 1);
    if (is_var_like_buffer && buffer_names_.count(name) != 0) {
      val_map.Set(name, op->value);
      return Stmt();
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    if (val_map.count(op->buffer->name)) {
      return val_map[op->buffer->name];
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }
  std::unordered_set<String> buffer_names_;
  Map<String, PrimExpr> val_map;
};

class VarSubstituter : public StmtExprMutator {
 public:
  explicit VarSubstituter() {}

 private:
  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_map_.find(GetRef<Var>(op));
    if (it != var_map_.end()) {
      return (*it).second;
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  Stmt VisitStmt_(const ForNode* op) final {
    For loop = GetRef<For>(op);
    loop.CopyOnWrite()->body = this->VisitStmt(op->body);
    return std::move(loop);
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    BlockRealize realize = GetRef<BlockRealize>(op);
    const auto* block_op = op->block.as<BlockNode>();
    for (size_t i = 0; i < op->iter_values.size(); ++i) {
      if (op->iter_values[i].as<VarNode>() == nullptr) {
        continue;
      }
      IterVar iter_var = block_op->iter_vars[i];
      Var loop_var = Downcast<Var>(op->iter_values[i]);
      Var v = iter_var->var;
      auto it = var_map_.find(loop_var);
      if (it == var_map_.end()) {
        var_map_.Set(loop_var, v);
      } else {
        LOG(FATAL) << "iter_var in BlockRealize has already defined";
        throw;
      }
    }
    realize.CopyOnWrite()->block = Downcast<Block>(this->VisitStmt(op->block));
    return std::move(realize);
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = GetRef<Block>(op);

    Array<IterVar> iter_vars;
    for (int i = 0; i < op->iter_vars.size(); ++i) {
      IterVar iter_var = op->iter_vars[i];
      PrimExpr min = iter_var->dom->min;
      PrimExpr extent = iter_var->dom->extent;
      iter_vars.push_back(
          IterVar(Range(min, extent), iter_var->var, iter_var->iter_type, iter_var->thread_tag));
    }

    Block n = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    BlockNode* block_ptr = n.CopyOnWrite();
    block_ptr->iter_vars = iter_vars;
    return std::move(n);
  }

  Map<Var, Var> var_map_;
};

class LetStmtInliner : public StmtExprMutator {
 public:
  explicit LetStmtInliner(Map<Var, PrimExpr> var_map = {}) : var_map_(var_map) {}

 private:
  Stmt VisitStmt_(const LetStmtNode* op) final {
    var_map_.Set(op->var, op->value);
    Stmt n = this->VisitStmt(op->body);
    var_map_.erase(op->var);
    return n;
  }
  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_map_.find(GetRef<Var>(op));
    if (it != var_map_.end()) {
      return this->VisitExpr(var_map_.at(GetRef<Var>(op)));
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = GetRef<Block>(op);
    Block n = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    BlockNode* block_ptr = n.CopyOnWrite();

    bool is_scope_root = (n->name_hint == "root");
    if (!is_scope_root) {
      AddBuffersInBlockSignature(n.get());
      Array<Array<BufferRegion>> inspected = GetBlockReadWriteRegion(n, buffer_var_map_);
      Array<BufferRegion> reads = std::move(inspected[0]);
      Array<BufferRegion> writes = std::move(inspected[1]);

      BufferCollector buffer_collector;
      buffer_collector(n);
      Array<Array<BufferRegion>> may_inspected_missed =
          GetBlockReadWriteRegion(n, buffer_collector.var2buffer_);
      Array<BufferRegion> may_reads_missed = std::move(may_inspected_missed[0]);
      Array<BufferRegion> may_writes_missed = std::move(may_inspected_missed[1]);

      MergeBufferRegions(reads, may_reads_missed);
      MergeBufferRegions(writes, may_writes_missed);
      block_ptr->reads = std::move(reads);
      block_ptr->writes = std::move(writes);
    }
    return std::move(n);
  }
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    BlockRealize realize = GetRef<BlockRealize>(op);
    const auto* block_op = op->block.as<BlockNode>();
    Map<Var, PrimExpr> replace_map;
    for (size_t i = 0; i < op->iter_values.size(); ++i) {
      if (op->iter_values[i].as<VarNode>() == nullptr) {
        continue;
      }
      IterVar iter_var = block_op->iter_vars[i];
      Var loop_var = Downcast<Var>(op->iter_values[i]);
      Var v = iter_var->var;
      auto it = var_map_.find(loop_var);
      if (it == var_map_.end()) {
        replace_map.Set(loop_var, v);
      } else {
        LOG(FATAL) << "iter_var in BlockRealize has already defined";
        throw;
      }
    }
    loop_var_map_.push_back(replace_map);
    realize.CopyOnWrite()->block = Downcast<Block>(this->VisitStmt(op->block));
    loop_var_map_.pop_back();
    return std::move(realize);
  }
  void AddBuffersInBlockSignature(const BlockNode* block) {
    for (const BufferRegion& buffer_region : block->reads) {
      const Buffer& buffer = buffer_region->buffer;
      buffer_var_map_.Set(buffer->data, buffer);
    }
    for (const BufferRegion& buffer_region : block->writes) {
      const Buffer& buffer = buffer_region->buffer;
      buffer_var_map_.Set(buffer->data, buffer);
    }
    for (const Buffer& buffer : block->alloc_buffers) {
      buffer_var_map_.Set(buffer->data, buffer);
    }
  }
  void MergeBufferRegions(Array<BufferRegion>& reads, const Array<BufferRegion>& may_reads_missed) {
    std::unordered_set<String, ObjectHash, ObjectEqual> existing_buffers;

    // 현재 reads에 있는 BufferRegion의 Buffer 정보를 저장
    for (const auto& br : reads) {
      existing_buffers.insert(br->buffer->name);
    }

    // may_reads_missed에서 reads에 없는 BufferRegion을 추가
    for (const auto& br : may_reads_missed) {
      if (existing_buffers.find(br->buffer->name) == existing_buffers.end()) {
        reads.push_back(br);
      }
    }
  }
  Map<Var, PrimExpr> var_map_;
  Array<Map<Var, PrimExpr>> loop_var_map_;
  Map<Var, Buffer> buffer_var_map_;
};

PrimFunc InlineLetStmt(PrimFunc func) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(func)) {
    auto fptr = func.CopyOnWrite();
    auto buffer_names = func->GetAttr<Array<String>>("private");
    if (buffer_names.defined()) {
      fptr->body = PrivateBufferInliner(buffer_names.value())(fptr->body);
    }
    fptr->body = MatchBufferInliner()(fptr->body);
    fptr->body = LetStmtInliner()(fptr->body);
    fptr->body = VarSubstituter()(fptr->body);
    return func;
  } else {
    return func;
  }
}

namespace transform {

Pass InlineLetStmt() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return InlineLetStmt(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InlineLetStmt", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InlineLetStmt").set_body_typed(InlineLetStmt);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
