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
#include <tvm/ir/op.h>
#include <tvm/tir/op.h>  // tir 관련 op들도 여기 들어있음
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

// 이미 있는 헤더들
#include <tvm/runtime/object.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
namespace tvm {
namespace tir {

// 루프 변수 교체를 위한 Mutator 정의

class StmtLoopVarReplacer : public tvm::tir::StmtExprMutator {
 public:
  Var old_var_;
  Var new_var_;

  StmtLoopVarReplacer(const Var& old_var, const Var& new_var)
      : old_var_(old_var), new_var_(new_var) {}

  tvm::PrimExpr VisitExpr_(const tvm::tir::VarNode* op) override {
    // 포인터 비교를 통해 기존 루프 변수와 동일한지 확인
    if (op == old_var_.get()) {
      return new_var_;
    }
    return tvm::tir::StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const tvm::tir::ForNode* op) override {
    // ForNode의 loop_var를 교체
    auto loop_var = (op->loop_var.get() == old_var_.get()) ? new_var_ : op->loop_var;

    // extent도 재귀적으로 처리
    auto min = this->VisitExpr(op->min);
    auto extent = this->VisitExpr(op->extent);  // extent를 변경할 수 있도록 추가

    // body를 재귀적으로 처리
    auto body = this->VisitStmt(op->body);

    // 새로운 ForNode 생성
    return tvm::tir::For(loop_var, min, extent, op->kind, body, op->thread_binding,
                         op->annotations);
  }
};

class BlockLoopVarReplacer : public StmtExprMutator {
 public:
  Var old_var_;
  Var new_var_;

  BlockLoopVarReplacer(const Var& old_var, const Var& new_var)
      : old_var_(old_var), new_var_(new_var) {}

  PrimExpr VisitExpr_(const VarNode* op) override {
    // 기존 루프 변수를 새로운 변수로 교체
    // std::cout << "Visiting VarNode: " << op->name_hint << std::endl;
    if (op == old_var_.get()) {
      // std::cout << "Replacing " << old_var_->name_hint << " with " << new_var_->name_hint <<
      // std::endl;
      return new_var_;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const ForNode* op) override {
    // ForNode 내부에서는 별도로 처리하지 않음
    // std::cout << "Visiting ForNode: " << op->loop_var->name_hint << std::endl;
    return StmtExprMutator::VisitStmt_(op);
  }
};

class PerfectlyNestedLoops : public StmtMutator {
 public:
  static PrimFunc Apply(PrimFunc func) {
    auto* n = func.CopyOnWrite();
    n->body = PerfectlyNestedLoops()(std::move(n->body));
    return func;
  }

 private:
  /*!
   * \brief Visit `For` nodes to enforce nested loops.
   */
  Stmt VisitStmt_(const ForNode* outer) final {
    Stmt body = this->VisitStmt(outer->body);
    // std::cout << "body: " << body << std::endl;
    // std::cout << std::endl;
    // If the body contains a sequence of statements, handle them
    if (const auto* seq = body.as<SeqStmtNode>()) {
      bool have_for = false;
      for (const auto& stmt : seq->seq) {
        if (stmt.as<ForNode>()) {
          have_for = true;
          break;
        } else if (stmt.as<LetStmtNode>()) {
          const auto* let_stmt_node = stmt.as<LetStmtNode>();
          // LetStmt의 body가 ForNode인지 확인
          if (let_stmt_node->body.as<ForNode>()) {
            have_for = true;
            break;
          }
          // std::cout << "stmt: " << stmt << std::endl;
        }
        // std::cout << "stmt: " << stmt << std::endl;
      }
      if (have_for) {
        return HandleSeqStmt(outer, seq);
      }
    }

    // Default case: return the loop with updated body
    auto n = CopyOnWrite(outer);
    n->body = body;
    return Stmt(n);
  }

  /*!
   * \brief Handle sequential statements inside a loop body.
   */
  Stmt HandleSeqStmt(const ForNode* outer, const SeqStmtNode* seq) {
    std::vector<Stmt> new_loops;

    for (const auto& stmt : seq->seq) {
      // If stmt is a Block or For, create a new loop
      Stmt new_stmt = stmt;
      if (stmt.as<BlockRealizeNode>()) {
        const auto* block_realize_node = stmt.as<BlockRealizeNode>();
        if (block_realize_node) {
          const auto* block = block_realize_node->block.as<BlockNode>();
          Map<Var, Var> record_new_iter_vars;
          if (block) {
            // 새로운 iter_vars 및 iter_values 생성
            Array<IterVar> new_iter_vars = block->iter_vars;
            Array<PrimExpr> new_iter_values = block_realize_node->iter_values;

            LOG(INFO) << "block->iter_vars.size(): " << block->iter_vars.size();

            // init이 없는 경우 처리
            Optional<Stmt> new_init = block->init;
            if (!new_init.defined()) {
              new_init = Evaluate(IntImm(DataType::Int(32), 0));
            }

            // 새로운 Block 생성
            Block new_block(new_iter_vars,         // 수정된 iter_vars
                            block->reads,          // 기존 reads 유지
                            block->writes,         // 기존 writes 유지
                            block->name_hint,      // 기존 이름 유지
                            block->body,           // 기존 body 유지
                            new_init,              // T.init 추가
                            block->alloc_buffers,  // 기존 alloc_buffers 유지
                            block->match_buffers,  // 기존 match_buffers 유지
                            block->annotations,    // 기존 annotations 유지
                            block->span            // 기존 span 유지
            );

            // std::cout << "new_block: " << new_block << std::endl;
            Block replaced_block = new_block;  // 일단 복사
            // std::cout << "replaced_block: " << replaced_block << std::endl;
            // 새로운 BlockRealize 생성
            auto new_block_realize =
                BlockRealize(new_iter_values,                // 수정된 iter_values
                             block_realize_node->predicate,  // 기존 predicate 유지
                             replaced_block                  // 수정된 Block
                );

            new_stmt = new_block_realize;
          }
          // std::cout << "new_stmt: " << new_stmt << std::endl;
        }

        new_loops.push_back(CreateNewForLoop(outer, new_stmt));
      } else if (stmt.as<ForNode>()) {
        new_loops.push_back(CreateNewForLoop(outer, stmt));
      } else if (stmt.as<LetStmtNode>()) {
        const auto* let_stmt_node = stmt.as<LetStmtNode>();
        // LetStmt의 body가 ForNode인지 확인
        if (let_stmt_node->body.as<ForNode>()) {
          const auto* for_node_in_let_body = let_stmt_node->body.as<ForNode>();
          // outer를 그대로 전달하여 새로운 For 루프 생성
          new_loops.push_back(CreateNewForLoop(outer, new_stmt));
          // LetStmt를 새로운 For 루프로 교체
        } else {
          // LetStmt의 body가 ForNode가 아니면 그대로 추가
          new_loops.push_back(stmt);
        }
      }

      else {
        // Otherwise, keep the statement as-is
        new_loops.push_back(stmt);
      }
    }

    // Return the sequence of new For loops
    return SeqStmt(new_loops);
  }

  /*!
   * \brief Create a new For loop wrapping a single statement.
   */

  Stmt CreateNewForLoop(const ForNode* outer, const Stmt& stmt) {
    // 기존 루프 변수와 구별되는 새로운 변수 생성
    // loop_var_new_name 에 있는지 확인
    // std::cout << "outer->loop_var->name_hint: " << outer->loop_var->name_hint << std::endl;
    if (loop_var_new_name.find(outer->loop_var->name_hint) == loop_var_new_name.end()) {
      loop_var_new_name[outer->loop_var->name_hint] = 0;
    } else {
      loop_var_new_name[outer->loop_var->name_hint] += 1;
    }

    Var new_loop_var(outer->loop_var->name_hint + "_" +
                         std::to_string(loop_var_new_name[outer->loop_var->name_hint]),
                     outer->loop_var->dtype);

    // 기존 루프 변수를 새로운 변수로 교체
    StmtLoopVarReplacer replacer(outer->loop_var, new_loop_var);
    Stmt updated_stmt = replacer(stmt);

    // 새로운 For 노드 생성
    return For(new_loop_var, outer->min, outer->extent, outer->kind, updated_stmt,
               outer->thread_binding, outer->annotations);
  }
  std::map<std::string, int> loop_var_new_name;
};

/*!
 * \brief Create a pass to enforce perfectly nested loops.
 */
namespace transform {
tvm::transform::Pass EnforcePerfectlyNestedLoops() {
  auto pass_func = [](PrimFunc func, IRModule mod, PassContext ctx) {
    return PerfectlyNestedLoops::Apply(std::move(func));
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0, "tir.EnforcePerfectlyNestedLoops",
                                                 {});
}

TVM_REGISTER_GLOBAL("tir.transform.EnforcePerfectlyNestedLoops")
    .set_body_typed(EnforcePerfectlyNestedLoops);
}  // namespace transform
}  // namespace tir
}  // namespace tvm