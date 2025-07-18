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
 * \file add_spatial_axis_to_loops.cc
 * \brief add_spatial_axis_to_loops
 */
#include <tvm/ir/op.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

/*!
 * \brief A Mutator to add T.axis.spatial to loops without T.axis,
 *        without creating new blocks.
 */

class LoopVarReplacer : public StmtExprMutator {
 public:
  static Stmt Replace(const Stmt& stmt, const Map<Var, Var>& var_replacement) {
    LoopVarReplacer replacer(var_replacement);
    return replacer(stmt);
  }

 private:
  explicit LoopVarReplacer(const Map<Var, Var>& var_replacement)
      : var_replacement_(var_replacement) {}

  PrimExpr VisitExpr_(const VarNode* op) override {
    auto it = var_replacement_.find(GetRef<Var>(op));
    if (it != var_replacement_.end()) {
      return (*it).second;  // Corrected: Use `*it` instead of `it->`
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const ForNode* op) override {
    // Do not replace variables inside nested loops
    return StmtExprMutator::VisitStmt_(op);
  }

  Map<Var, Var> var_replacement_;
};
class AddAxisWithLoops : public StmtMutator {
 public:
  static PrimFunc Apply(PrimFunc func) {
    auto* n = func.CopyOnWrite();
    n->body = AddAxisWithLoops()(std::move(n->body));
    return func;
  }

 private:
  // Stack to track loop variables in nested loops
  std::vector<std::pair<Var, Range>> loop_var_stack;

  Stmt VisitStmt_(const ForNode* loop) override {
    LOG(INFO) << "[VisitStmt_] Entering ForNode: " << loop->loop_var->name_hint;

    // Push the current loop variable and its range (min, extent) to the stack
    loop_var_stack.emplace_back(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));

    // Visit the loop body
    Stmt new_body = StmtMutator::VisitStmt(loop->body);

    // Check if the body contains a T.axis or BlockRealize
    auto block_axis_map = ContainsAxisPerBlock(loop->body);
    bool has_axis = std::all_of(block_axis_map.begin(), block_axis_map.end(),
                                [](const auto& pair) { return pair.second; });
    LOG(INFO) << "[VisitStmt_] Result of ContainsAxisPerBlock for " << loop->loop_var->name_hint
              << ": " << (has_axis ? "true" : "false");

    if (!has_axis) {
      LOG(INFO) << "[VisitStmt_] No T.axis found in body of " << loop->loop_var->name_hint;

      // Add T.axis.spatial for all variables in the stack
      if (const auto* block_realize = new_body.as<BlockRealizeNode>()) {
        const auto* block = block_realize->block.as<BlockNode>();
        if (block) {
          LOG(INFO) << "[VisitStmt_] Processing Block: " << block->name_hint;

          Array<IterVar> new_iter_vars(block->iter_vars);
          Array<PrimExpr> new_iter_values(block_realize->iter_values);
          Map<Var, Var> var_replacement;

          // Add T.axis.spatial for all loop variables in the stack
          for (const auto& [loop_var, range] : loop_var_stack) {
            std::string iter_var_name = loop_var->name_hint + "_it";
            Var iter_var_var(iter_var_name, loop_var->dtype);

            LOG(INFO) << "[VisitStmt_] Adding T.axis.spatial for loop_var: " << iter_var_name;

            auto iter_var = IterVar(range,  // Use the stored range (min, extent)
                                    iter_var_var, IterVarType::kDataPar);

            // Avoid duplicate IterVars
            if (std::none_of(new_iter_vars.begin(), new_iter_vars.end(),
                             [&](const IterVar& iv) { return iv->var.get() == loop_var.get(); })) {
              new_iter_vars.push_back(iter_var);
              new_iter_values.push_back(loop_var);

              // Store the replacement for later use
              var_replacement.Set(loop_var, iter_var_var);
            }
          }

          // Replace variables in the block using LoopVarReplacer
          Block replaced_block =
              Downcast<Block>(LoopVarReplacer::Replace(GetRef<Stmt>(block), var_replacement));

          if (!replaced_block->init.defined()) {
            LOG(INFO) << "[VisitStmt_] Block has no init. Setting init to T.evaluate(1).";
            replaced_block.CopyOnWrite()->init =
                tvm::tir::Evaluate(tvm::IntImm(DataType::Int(32), 1));
          }
          //   replaced_block.CopyOnWrite()->reads = Array<BufferRegion>();
          // replaced_block.CopyOnWrite()->writes = Array<BufferRegion>();
          // Debugging: Check iter_vars and iter_values size
          LOG(INFO) << "[VisitStmt_] iter_vars size: " << new_iter_vars.size();
          LOG(INFO) << "[VisitStmt_] iter_values size: " << new_iter_values.size();

          // Create a new Block with updated iter_vars
          replaced_block.CopyOnWrite()->iter_vars = new_iter_vars;

          // Create a new BlockRealize with updated iter_values and replaced block
          auto new_block_realize =
              BlockRealize(new_iter_values, block_realize->predicate, replaced_block);

          LOG(INFO) << "[VisitStmt_] Updated BlockRealize created with all loop_vars.";

          // Wrap the updated BlockRealize in the original loop
          auto n = CopyOnWrite(loop);
          n->body = new_block_realize;

          // Pop the current loop variable and its range from the stack
          loop_var_stack.pop_back();

          return Stmt(n);
        }
      }
      // seq_stmt
      //  Handle SeqStmtNode
      else if (const auto* seq_stmt = new_body.as<SeqStmtNode>()) {
        LOG(INFO) << "[VisitStmt_] Processing SeqStmt";
        Array<Stmt> new_stmts;

        for (const auto& stmt : seq_stmt->seq) {
          if (const auto* block_realize = stmt.as<BlockRealizeNode>()) {
            const auto* block = block_realize->block.as<BlockNode>();
            if (block && !block_axis_map[block]) {
              LOG(INFO) << "[VisitStmt_] Block " << block->name_hint
                        << " does not contain T.axis. Adding axis.";

              Array<IterVar> new_iter_vars(block->iter_vars);
              Array<PrimExpr> new_iter_values(block_realize->iter_values);
              Map<Var, Var> var_replacement;

              for (const auto& [loop_var, range] : loop_var_stack) {
                std::string iter_var_name = loop_var->name_hint + "_it";
                Var iter_var_var(iter_var_name, loop_var->dtype);

                auto iter_var = IterVar(range, iter_var_var, IterVarType::kDataPar);

                if (std::none_of(
                        new_iter_vars.begin(), new_iter_vars.end(),
                        [&](const IterVar& iv) { return iv->var.get() == loop_var.get(); })) {
                  new_iter_vars.push_back(iter_var);
                  new_iter_values.push_back(loop_var);
                  var_replacement.Set(loop_var, iter_var_var);
                }
              }

              Block replaced_block =
                  Downcast<Block>(LoopVarReplacer::Replace(GetRef<Stmt>(block), var_replacement));

              if (!replaced_block->init.defined()) {
                replaced_block.CopyOnWrite()->init =
                    tvm::tir::Evaluate(tvm::IntImm(DataType::Int(32), 1));
              }
              //   replaced_block.CopyOnWrite()->reads = Array<BufferRegion>();
              replaced_block.CopyOnWrite()->iter_vars = new_iter_vars;

              auto new_block_realize =
                  BlockRealize(new_iter_values, block_realize->predicate, replaced_block);

              LOG(INFO) << "[VisitStmt_] Updated BlockRealize created for SeqStmt.";
              new_stmts.push_back(new_block_realize);
            } else {
              // T.axis.spatial already exists, keep the block as is
              LOG(INFO) << "[VisitStmt_] Block " << block->name_hint << " already contains T.axis.";
              new_stmts.push_back(stmt);
            }
          } else {
            // Recursively visit other statements
            new_stmts.push_back(StmtMutator::VisitStmt(stmt));
          }
        }

        auto n = CopyOnWrite(loop);
        n->body = SeqStmt(new_stmts);
        loop_var_stack.pop_back();
        return Stmt(n);
      }
    }

    // Pop the current loop variable and its range from the stack
    loop_var_stack.pop_back();

    // Return the original loop if no changes are made
    auto n = CopyOnWrite(loop);
    n->body = std::move(new_body);
    return Stmt(n);
  }

  std::unordered_map<const BlockNode*, bool> ContainsAxisPerBlock(const Stmt& stmt) {
    struct AxisChecker : public StmtVisitor {
      std::unordered_map<const BlockNode*, bool> block_axis_map;

      void VisitStmt_(const BlockRealizeNode* node) override {
        const auto* block = node->block.as<BlockNode>();
        if (block) {
          // Block의 iter_vars 유무를 판단하여 저장
          bool has_axis = !block->iter_vars.empty();
          block_axis_map[block] = has_axis;

          LOG(INFO) << "[ContainsAxisPerBlock] BlockRealize " << block->name_hint
                    << " has_axis: " << has_axis;

          for (const auto& iter_var : block->iter_vars) {
            LOG(INFO) << "[ContainsAxisPerBlock] iter_var: " << iter_var->var->name_hint;
          }
        }
        StmtVisitor::VisitStmt_(node);  // Continue visiting
      }

      void VisitStmt_(const BlockNode* node) override {
        LOG(INFO) << "[ContainsAxisPerBlock] Visiting BlockNode: " << node->name_hint;

        // Block의 iter_vars 유무를 판단하여 저장
        bool has_axis = !node->iter_vars.empty();
        block_axis_map[node] = has_axis;

        if (has_axis) {
          for (const auto& iter_var : node->iter_vars) {
            LOG(INFO) << "[ContainsAxisPerBlock] iter_var: " << iter_var->var->name_hint;
          }
        } else {
          LOG(INFO) << "[ContainsAxisPerBlock] BlockNode has no iter_vars.";
        }
        StmtVisitor::VisitStmt_(node);  // Continue visiting
      }

      void VisitStmt_(const ForNode* node) override {
        LOG(INFO) << "[ContainsAxisPerBlock] Visiting ForNode: " << node->loop_var->name_hint;
        StmtVisitor::VisitStmt(node->body);  // Check nested loops
      }
    };

    AxisChecker checker;
    checker(stmt);

    // Debugging: Map 결과 출력
    for (const auto& pair : checker.block_axis_map) {
      LOG(INFO) << "[ContainsAxisPerBlock] Block " << pair.first->name_hint
                << " has_axis: " << pair.second;
    }

    return checker.block_axis_map;
  }
};
/*!
 * \brief Create a pass to add T.axis.spatial to loops without T.axis,
 *        without creating new blocks.
 */
namespace transform {
tvm::transform::Pass AddSpatialAxisToLoopsPass() {
  auto pass_func = [](PrimFunc func, IRModule mod, PassContext ctx) {
    return AddAxisWithLoops::Apply(std::move(func));
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0, "tir.AddSpatialAxisToLoopsPass", {});
}

TVM_REGISTER_GLOBAL("tir.transform.AddSpatialAxisToLoopsPass")
    .set_body_typed(AddSpatialAxisToLoopsPass);
}  // namespace transform
}  // namespace tir
}  // namespace tvm