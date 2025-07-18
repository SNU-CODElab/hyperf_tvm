
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
 * \file perfectly_nested_loop.cc
 * \brief perfectly_nested_loop
 */
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

namespace tvm {
namespace tir {

struct RangeInfo {
  bool valid;
  int min_val;
  int max_val;
};

/*!
 * \brief Var에 대해 "loop_bounds_에서 (min_expr, extent_expr)"를 찾아
 *        실제 min_int, max_int를 구한다. (예: min=0, extent=1024 → max=1023)
 */
inline RangeInfo GetMinMaxOfVar(
    const VarNode* var_node,
    const std::unordered_map<const VarNode*, std::pair<PrimExpr, PrimExpr>>& loop_bounds) {
  auto it = loop_bounds.find(var_node);
  if (it == loop_bounds.end()) {
    // Var가 loop_bounds에 없으면 실패
    return {false, 0, 0};
  }
  // it->second = (min_expr, extent_expr)
  PrimExpr min_expr = it->second.first;
  PrimExpr ext_expr = it->second.second;

  // 둘 다 IntImm일 때만 간단히 처리
  if (auto min_imm = min_expr.as<IntImmNode>()) {
    if (auto ext_imm = ext_expr.as<IntImmNode>()) {
      int mn = static_cast<int>(min_imm->value);
      int ex = static_cast<int>(ext_imm->value);
      // 최대값 = min + extent - 1
      return {true, mn, mn + ex - 1};
    }
  }
  // 그 외 케이스는 여기서는 단순 처리 (실패)
  return {false, 0, 0};
}

/*!
 * \brief 주어진 expr에 대해 (가능하면) [min_val, max_val]를 추론.
 *        실패하면 RangeInfo.valid = false.
 *
 * loop_bounds: VarNode -> (min_expr, extent_expr) 형태
 */
RangeInfo InferRange(
    const PrimExpr& expr,
    const std::unordered_map<const VarNode*, std::pair<PrimExpr, PrimExpr>>& loop_bounds) {
  // 1) 상수 IntImm
  if (const IntImmNode* imm = expr.as<IntImmNode>()) {
    int v = static_cast<int>(imm->value);
    return {true, v, v};
  }

  // 2) Var
  else if (const VarNode* var_node = expr.as<VarNode>()) {
    return GetMinMaxOfVar(var_node, loop_bounds);
  }

  // 3) Add
  else if (const AddNode* add_node = expr.as<AddNode>()) {
    RangeInfo left = InferRange(add_node->a, loop_bounds);
    RangeInfo right = InferRange(add_node->b, loop_bounds);
    if (!left.valid || !right.valid) {
      return {false, 0, 0};
    }
    // 최대값 = left.max_val + right.max_val
    // 최소값 = left.min_val + right.min_val
    return {true, left.min_val + right.min_val, left.max_val + right.max_val};
  }

  // 4) Sub
  else if (const SubNode* sub_node = expr.as<SubNode>()) {
    RangeInfo left = InferRange(sub_node->a, loop_bounds);
    RangeInfo right = InferRange(sub_node->b, loop_bounds);
    if (!left.valid || !right.valid) {
      return {false, 0, 0};
    }
    // 최대값 = left.max_val - right.min_val
    // 최소값 = left.min_val - right.max_val
    int mn = left.min_val - right.max_val;
    int mx = left.max_val - right.min_val;
    return {true, mn, mx};
  }

  // 5) Mul (단순 가정: 둘 다 비음수라면 최대= left.max * right.max)
  else if (const MulNode* mul_node = expr.as<MulNode>()) {
    RangeInfo left = InferRange(mul_node->a, loop_bounds);
    RangeInfo right = InferRange(mul_node->b, loop_bounds);
    if (!left.valid || !right.valid) {
      return {false, 0, 0};
    }
    // 최소값, 최대값 계산 (음수 가능성은 무시한 간단 버전)
    int mn = left.min_val * right.min_val;
    int mx = left.max_val * right.max_val;
    return {true, mn, mx};
  }

  // 그 외는 여기서는 처리 X
  return {false, 0, 0};
}

/*!
 * \brief "expr이 loop_bounds를 기반으로 상수범위로 평가 가능하면" => (true, 그 최대값)
 *        아니면 (false, 0)
 *
 *        내부적으로는 InferRange로 [min, max]를 구하고, 그중 max를 반환
 */
std::pair<bool, int> IsConstantOrLoopBound(
    const PrimExpr& expr,
    const std::unordered_map<const VarNode*, std::pair<PrimExpr, PrimExpr>>& loop_bounds) {
  RangeInfo r = InferRange(expr, loop_bounds);
  if (r.valid) {
    return {true, r.max_val};
  } else {
    return {false, 0};
  }
}

class LoopConditionCollector : public StmtMutator {
 public:
  // ForNode를 후순위(post-order)로 방문:
  //   1) body 먼저 VisitStmt -> 재귀 하위 변환/수집
  //   2) 현재 ForNode의 extent가 상수/loop_bound로 표현 가능한지 검사
  Stmt VisitStmt_(const ForNode* op) override {
    // (A) 현재 루프 바운드 정보 저장
    loop_bounds_[op->loop_var.get()] = {op->min, op->extent};
    LOG(INFO) << "Loop variable: " << op->loop_var->name_hint;
    LOG(INFO) << "Loop min: " << op->min;
    LOG(INFO) << "Loop extent: " << op->extent;

    // (B) body를 재귀 방문 -> 변환된 body
    Stmt body = StmtMutator::VisitStmt(op->body);

    // (C) extent가 상수(또는 현재까지의 loop_bounds 내에서 계산 가능한 식)인지 확인
    if (!op->extent.as<IntImmNode>()) {
      auto result = IsConstantOrLoopBound(op->extent, loop_bounds_);
      bool is_valid_extent = result.first;
      int max_extent = result.second;

      if (!is_valid_extent) {
        LOG(WARNING) << "Extent is not constant or dependent on loop bounds.";
      } else {
        LOG(INFO) << "Extent is valid: dependent only on loop bounds. Max value: " << max_extent;
        PrimExpr bool_cond = tvm::tir::LT(op->loop_var, op->extent);
        loop_conditions_[op->loop_var.get()] = bool_cond;
        LOG(INFO) << "Loop condition: " << bool_cond;
        // extent를 상수로 대체 가능하다면 새 For 노드 생성
        if (is_valid_extent) {
          auto new_extent = tvm::IntImm(DataType::Int(32), max_extent);
          // (D) 스코프에서 빠지기 전에(필요시) loop_bounds_를 업데이트하거나 유지
          //     여기서는 수집만 하고 그대로 둔다고 가정
          return For(op->loop_var, op->min, new_extent, op->kind, body, op->thread_binding,
                     op->annotations);
        } else {
          LOG(FATAL) << "max_extent is not an IntImmNode: " << max_extent;
        }
      }
    }
    // extent에 변화가 없다면 원본 그대로 반환
    return For(op->loop_var, op->min, op->extent, op->kind, body, op->thread_binding,
               op->annotations);
  }

  // (필요하다면 다른 노드들(Block, IfThenElse 등)도 오버라이드)
  // 여기서는 ForNode만 특별 처리한다고 가정

  // Collector가 끝난 뒤, 각 루프Var -> (min, extent)와 condition(사용된 식)을 갖고 있음
  std::unordered_map<const VarNode*, std::pair<PrimExpr, PrimExpr>> loop_bounds_;
  std::unordered_map<const VarNode*, PrimExpr> loop_conditions_;
};

class LoopConditionReplacer : public StmtMutator {
 public:
  explicit LoopConditionReplacer(std::unordered_map<const VarNode*, PrimExpr> conditions)
      : loop_conditions_(std::move(conditions)) {}

  // ForNode 등 필요한 노드들은 전부 자식 먼저 방문
  Stmt VisitStmt_(const ForNode* op) override {
    LOG(INFO) << "visit For";
    Stmt new_body = VisitStmt(op->body);
    return For(op->loop_var, op->min, op->extent, op->kind, new_body, op->thread_binding,
               op->annotations);
  }

  // BlockNode도 재귀방문(필요하다면)
  Stmt VisitStmt_(const BlockNode* op) override {
    LOG(INFO) << "visit Block";
    Stmt new_body = VisitStmt(op->body);
    return Block(op->iter_vars, op->reads, op->writes, op->name_hint, new_body, op->init,
                 op->alloc_buffers, op->match_buffers, op->annotations, op->span);
  }

  //---- [중요!] BlockRealizeNode에서 predicate를 수정해야 T.where가 생김
  Stmt VisitStmt_(const BlockRealizeNode* op) override {
    // 1) 먼저 하위 노드(즉 BlockNode)를 방문해서 변환
    LOG(INFO) << "visit BlockRealize";
    Block new_block = Downcast<Block>(VisitStmt(op->block));

    // 2) 기존 predicate
    PrimExpr new_predicate = op->predicate;

    for (auto PE : op->iter_values) {
      if (const auto* var_node = PE.as<tvm::tir::VarNode>()) {
        // loop_conditions_에서 조건 확인
        auto it = loop_conditions_.find(var_node);
        if (it != loop_conditions_.end()) {
          LOG(INFO) << "Found condition for iter value: " << var_node->name_hint;
          new_predicate = new_predicate && it->second;
        }
      } else {
        LOG(WARNING) << "Unexpected iter_value type, not a Var: " << var_node->name_hint;
      }
    }
    // 4) 새 BlockRealize 반환 (이 predicate가 T.where(...)에 해당)
    return BlockRealize(op->iter_values, new_predicate, new_block);
  }

 private:
  // var_name -> var<extent> 형태의 bool 식
  std::unordered_map<const VarNode*, PrimExpr> loop_conditions_;
};

PrimFunc Apply(PrimFunc func) {
  auto* n = func.CopyOnWrite();

  // (1) Collector로 먼저 loop_conditions_를 수집하고,
  //     For 노드를 만나면 extent를 상수화할 수도 있음.
  LoopConditionCollector collector;
  // -- 꼭 **결과**를 받아야 한다! --
  //    collector(...)는 변환된 stmt를 반환하므로 n->body에 다시 넣어주어야 함.
  Stmt collected = collector(n->body);  // post-order로 수집 & 변경
  n->body = collected;

  // (2) 수집한 loop_conditions_를 이용해 BlockNode에 IfThenElse를 삽입
  LoopConditionReplacer replacer(collector.loop_conditions_);
  Stmt replaced = replacer(n->body);
  n->body = replaced;

  return func;
}

namespace transform {

tvm::transform::Pass SimplifyLoopBounds() {
  auto pass_func = [](PrimFunc func, IRModule mod, PassContext ctx) {
    return Apply(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.SimplifyLoopBounds", {});
}

TVM_REGISTER_GLOBAL("tir.transform.SimplifyLoopBounds").set_body_typed(SimplifyLoopBounds);

}  // namespace transform
}  // namespace tir
}  // namespace tvm