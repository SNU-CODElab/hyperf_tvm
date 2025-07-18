#include <tvm/tir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/op.h>
#include <tvm/ir/op.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace tir {

class ExpandReductionBuffersMutator : public StmtExprMutator {
 public:
  explicit ExpandReductionBuffersMutator(PrimFunc func, std::vector<std::string> shared_loop_vars) : func_(func) ,shared_loop_vars_(shared_loop_vars) {
    for (const auto& kv : func->buffer_map) {
      buffers_.insert(kv.second->name);
      LOG(INFO) << "Buffer found in func: " << kv.second->name;
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) override {
    const BlockNode* block = op->block.get();
    LOG(INFO) << "Entering BlockRealize: " << block->name_hint;

    if (block->name_hint == "root") {
      LOG(INFO) << "Processing root block for buffer allocation.";
      Stmt new_body = StmtExprMutator::VisitStmt(block->body);

      // Add the new buffer to alloc_buffers
      Array<Buffer> new_alloc_buffers = block->alloc_buffers;
      new_alloc_buffers.push_back(buffer_alias_map_[target_buffer_]);

      // Reconstruct the root block
      Block new_block = Block(
          block->iter_vars,
          block->reads,
          block->writes,
          block->name_hint,
          new_body,
          block->init,
          new_alloc_buffers,  // Add the new buffer
          block->match_buffers,
          block->annotations,
          block->span
      );

      // Reconstruct BlockRealize with updated Block
      return BlockRealize(op->iter_values, op->predicate, new_block);
    }

    bool is_reduction_block = false;

    // Check if this block is a reduction block
    for (const auto& iter_var : block->iter_vars) {
      if (iter_var->iter_type == kCommReduce) {
        is_reduction_block = true;
        break;
      }
    }

    bool has_valid_buffer = false;

    for (const auto& buffer_region : block->writes) {
      const Buffer& buffer = buffer_region->buffer;
      LOG(INFO) << "Checking buffer: " << buffer->name;

      // Calculate the total size of the buffer
      int64_t total_size = 1;
      bool is_valid_size = true;

      for (const auto& dim : buffer->shape) {
        const auto* int_dim = dim.as<IntImmNode>();
        if (int_dim) {
          total_size *= int_dim->value;
        } else {
          is_valid_size = false;  // Non-constant shape, size is unknown
          break;
        }
      }

      // if (is_valid_size && total_size == 1) {
      if (is_valid_size) {
        has_valid_buffer = true;
        LOG(INFO) << "Valid reduction buffer found: " << buffer->name;
        break;  // Only need to confirm one valid buffer
      } else if (!is_valid_size) {
        LOG(INFO) << "Skipping buffer due to unknown size: " << buffer->name;
      } else {
        LOG(INFO) << "Skipping buffer with size != 1: " << buffer->name;
      }
    }

    Optional<Stmt> new_init = block->init;
    Array<BufferRegion> new_reads = block->reads;
    Array<BufferRegion> new_writes = block->writes;



    if (is_reduction_block && has_valid_buffer) {
      LOG(INFO) << "Reduction block detected: " << block->name_hint;

      for (const auto& buffer_region : block->writes) {
        const Buffer& buffer = buffer_region->buffer;
        LOG(INFO) << "Checking buffer: " << buffer->name;

        if (buffers_.count(buffer->name)) {
          LOG(INFO) << "Target buffer found: " << buffer->name;
          target_buffer_ = buffer;  // Set target_buffer_

          index_pattern_ = CreateIndexPatternFromLoopVars();
          break;  // Handle the first target buffer only
        }
      }
      Var data_var = Var("reduction_buffer", PointerType(PrimType(target_buffer_->dtype)));
      //shape target_buffer_
      Array<PrimExpr> shape;
      for (const auto& dim : target_buffer_->shape) {
        shape.push_back(dim);
      }


      for (const auto& loop_var : index_pattern_) {
        if (const auto* var_node = loop_var.as<tvm::tir::VarNode>()) {
          shape.push_back(size_map_[var_node->name_hint]);
        } else {
          LOG(FATAL) << "Expected a VarNode but got: " << loop_var->GetTypeKey();
        }
      }
      LOG(INFO) << "Expanded shape: " << shape;
      Buffer new_buffer = Buffer(
          data_var,                                // 새 포인터 변수
          target_buffer_->dtype,                  // 기존 데이터 타입 유지
          shape,                         // 새 Shape
          {},                                     // Strides 비워둠
          PrimExpr(),                             // Element offset 비워둠
          target_buffer_->name + "_expanded",     // "_expanded" 추가한 이름
          target_buffer_->data_alignment,         // 기존 정렬 유지
          target_buffer_->offset_factor,          // 기존 offset factor 유지
          BufferType::kDefault                    // 기본 Buffer type
      );

      // 기존 Buffer와 새 Buffer 매핑
      buffer_alias_map_[target_buffer_] = new_buffer;
      LOG(INFO) << "Mapped buffer " << target_buffer_->name << " to " << new_buffer->name;
    }

  
    for (size_t i = 0; i < op->iter_values.size(); ++i) {
      block_iter_var_map_[op->iter_values[i]] = block->iter_vars[i]->var;
      LOG(INFO) << "Mapped iter_value: " << op->iter_values[i]
                << " to iter_var: " << block->iter_vars[i]->var->name_hint;
    }
    Stmt new_body = StmtExprMutator::VisitStmt(block->body);

    if (block->init.defined()) {
      new_init = StmtExprMutator::VisitStmt(block->init.value());
      LOG(INFO) << "Adjusted init block for reduction.";
    }

   for (size_t i = 0; i < block->reads.size(); ++i) {
      const BufferRegion& read = block->reads[i];
      if (buffer_alias_map_.count(read->buffer)) {
          // 원래 접근하는 인덱스 추출 (read->region에서 min 값 사용)
          Array<PrimExpr> original_indices;
          for (const auto& range : read->region) {
              original_indices.push_back(range->min);
          }

          // 기존 AdjustIndices 유지
          auto adjusted_indices = AdjustIndices(index_pattern_);

          // 원래 인덱스를 AdjustIndices 결과 앞에 추가
          Array<PrimExpr> final_indices;
          final_indices.insert(final_indices.end(), original_indices.begin(), original_indices.end());
          final_indices.insert(final_indices.end(), adjusted_indices.begin(), adjusted_indices.end());

          LOG(INFO) << "Original indices for read: " << original_indices;
          LOG(INFO) << "Adjusted indices: " << adjusted_indices;
          LOG(INFO) << "Final adjusted indices for read: " << final_indices;

          Array<Range> adjusted_region = ConvertToRange(final_indices);
          new_reads.Set(i, BufferRegion(buffer_alias_map_[read->buffer], adjusted_region));
      } 
    }

    for (size_t i = 0; i < block->writes.size(); ++i) {
        const BufferRegion& write = block->writes[i];
        if (buffer_alias_map_.count(write->buffer)) {
            // 원래 접근하는 인덱스 추출 (write->region에서 min 값 사용)
            Array<PrimExpr> original_indices;
            for (const auto& range : write->region) {
                original_indices.push_back(range->min);
            }

            // 기존 AdjustIndices 유지
            auto adjusted_indices = AdjustIndices(index_pattern_);

            // 원래 인덱스를 AdjustIndices 결과 앞에 추가
            Array<PrimExpr> final_indices;
            final_indices.insert(final_indices.end(), original_indices.begin(), original_indices.end());
            final_indices.insert(final_indices.end(), adjusted_indices.begin(), adjusted_indices.end());

            LOG(INFO) << "Original indices for write: " << original_indices;
            LOG(INFO) << "Adjusted indices: " << adjusted_indices;
            LOG(INFO) << "Final adjusted indices for write: " << final_indices;

            Array<Range> adjusted_region = ConvertToRange(final_indices);
            new_writes.Set(i, BufferRegion(buffer_alias_map_[write->buffer], adjusted_region));
        } 
    }

    for (size_t i = 0; i < op->iter_values.size(); ++i) {
      block_iter_var_map_.erase(op->iter_values[i]);
      LOG(INFO) << "Removed mapping for iter_value: " << op->iter_values[i];
    }

    Block new_block = Block(
        block->iter_vars,
        new_reads,
        new_writes,
        block->name_hint,
        new_body,
        new_init,
        block->alloc_buffers,
        block->match_buffers,
        block->annotations,
        block->span
    );

    return BlockRealize(op->iter_values, op->predicate, new_block);
  }

  Stmt VisitStmt_(const LetStmtNode* op) override {
    LOG(INFO) << "Visiting LetStmt: " << op->var->name_hint;

    // Map the LetStmt's var to its value
    let_var_map_[op->var] = op->value;
    LOG(INFO) << "Stored LetStmt var: " << op->var->name_hint
              << " with value: " << op->value;

    // Visit the body of the LetStmt
    Stmt new_body = StmtExprMutator::VisitStmt(op->body);

    // Reconstruct the LetStmt
    return LetStmt(op->var, op->value, new_body, op->span);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) override {
    if (op->buffer.same_as(target_buffer_)) {
      LOG(INFO) << "BufferStoreNode for target buffer: " << target_buffer_->name;

      Array<PrimExpr> new_index_pattern = op->indices;
      for (const auto& index : index_pattern_) {
        new_index_pattern.push_back(index);
      }

      Array<PrimExpr> new_indices;
      for (const auto& index : AdjustIndices(new_index_pattern)) {
        new_indices.push_back(index);
      }
      

      PrimExpr new_value = this->VisitExpr(op->value);

      return BufferStore(buffer_alias_map_[target_buffer_], new_value, new_indices);
    }

    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    LOG(INFO) << "BufferLoadNode for buffer: " << op->buffer->name;
    if (op->buffer.same_as(target_buffer_)) {
      LOG(INFO) << "BufferLoadNode for target buffer: " << target_buffer_->name;
      
      Array<PrimExpr> new_index_pattern = op->indices;
      for (const auto& index : index_pattern_) {
        new_index_pattern.push_back(index);
      }

      Array<PrimExpr> new_indices;
      for (const auto& index : AdjustIndices(new_index_pattern)) {
        new_indices.push_back(index);
      }
      

      return BufferLoad(buffer_alias_map_[target_buffer_], new_indices);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const ForNode* op) override {
    LOG(INFO) << "Entering For loop: " << op->loop_var->name_hint
              << " at address: " << op->loop_var.get();

    // Push the loop_var onto the stack
    loop_vars_.push_back(op->loop_var);
    PrimExpr extent = op->extent;

    // Check if the extent is mapped by a LetStmt
    if (const VarNode* var_node = extent.as<VarNode>()) {
      auto it = let_var_map_.find(GetRef<Var>(var_node));
      if (it != let_var_map_.end()) {
        extent = it->second;
        LOG(INFO) << "Using extent from LetStmt for loop_var: "
                  << op->loop_var->name_hint << " -> " << extent;
      }
    } else {
      LOG(INFO) << "Extent is not a VarNode, using original extent.";
    }

    // Store the loop_var and extent in size_map_
    size_map_[op->loop_var->name_hint] = extent;

    Stmt new_body = StmtExprMutator::VisitStmt(op->body);

    // Update the For loop body
    auto for_node = make_object<ForNode>(*op);
    for_node->body = new_body;

    // Pop the loop_var from the stack
    loop_vars_.pop_back();
    LOG(INFO) << "Exiting For loop: " << op->loop_var->name_hint;

    return Stmt(for_node);
  }

 private:
  PrimFunc func_;
  std::unordered_set<std::string> buffers_;
  Buffer target_buffer_;
  Array<PrimExpr> index_pattern_;
  std::map<std::string, PrimExpr> size_map_;
  std::vector<Var> loop_vars_;
  std::unordered_map<PrimExpr, Var, ObjectPtrHash, ObjectPtrEqual> block_iter_var_map_;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> let_var_map_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_alias_map_;
  std::vector<std::string> shared_loop_vars_;

  Array<PrimExpr> AdjustIndices(const Array<PrimExpr>& indices) {
    Array<PrimExpr> new_indices;

    for (const auto& index : indices) {
      auto it = block_iter_var_map_.find(index);  // Map iter_value to iter_var
      if (it != block_iter_var_map_.end()) {
        new_indices.push_back(it->second);
        LOG(INFO) << "Using Block iter_var: " << it->second->name_hint
                  << " for iter_value: " << index;
      } else {
        new_indices.push_back(index);
        LOG(INFO) << "Retaining original index: " << index;
      }
    }

    return new_indices;
  }

  Array<PrimExpr> CreateIndexPatternFromLoopVars() {
    Array<PrimExpr> pattern;

    for (size_t i = 0; i < loop_vars_.size() - 1; ++i) {
      if (std::find(shared_loop_vars_.begin(), shared_loop_vars_.end(), loop_vars_[i]->name_hint) != shared_loop_vars_.end()) {
        pattern.push_back(loop_vars_[i]);
        LOG(INFO) << "Adding shared loop variable to index pattern: " << loop_vars_[i]->name_hint;
      }
    }

    return pattern;
  }

  Array<Range> ConvertToRange(const Array<PrimExpr>& indices) {
    Array<Range> ranges;
    for (const auto& index : indices) {
        // Range(min, extent)
        LOG(INFO) << "Creating range from index: " << index;
        ranges.push_back(Range::FromMinExtent(index, 1));
    }
    return ranges;
  }
};


class CheckReductionReadUsage : public StmtExprVisitor {
 public:
  /*!
   * \brief Check if any buffer written in a reduction block is read in subsequent blocks.
   */
  bool IsReadAfterReduction() const { return is_read_after_reduction_; }

  /*!
   * \brief Get shared loop variables between write and read blocks.
   */
  std::vector<std::string> GetSharedLoopVars() const { return shared_loop_vars_; }

  void FindSharedLoopVars() {
      shared_loop_vars_.clear();

      // read_loop_vars_를 set으로 변환하여 빠르게 조회할 수 있도록 함
      std::unordered_set<std::string> read_loop_vars_set(read_loop_vars_.begin(), read_loop_vars_.end());

      for (const auto& var : write_loop_vars_) {
          if (read_loop_vars_set.count(var)) {  // write_loop_vars_에 있는 변수 중 read에도 있는 경우 추가
              shared_loop_vars_.push_back(var);
          }
      }

      LOG(INFO) << "Shared loop variables: ";
      for (const auto& var : shared_loop_vars_) {
          LOG(INFO) << var;
      }
  }


 private:
  void VisitStmt_(const BlockRealizeNode* op) final {
    const BlockNode* block = op->block.get();

    // Check if this block is a reduction block
    bool is_reduction_block = false;
    for (const auto& iter_var : block->iter_vars) {
        if (iter_var->iter_type == kCommReduce) {
            is_reduction_block = true;
            break;
        }
    }

    if (is_reduction_block) {
        LOG(INFO) << "Reduction block detected: " << block->name_hint;
        
        // Reduction Block에서 쓰는 루프 변수 저장
        if (!loop_vars_.empty()) {  // 빈 경우 예외 방지
            for (auto it = loop_vars_.begin(); it != loop_vars_.end(); ++it) {
                // 없는 경우만 추가 (std::find 사용)
                if (std::find(write_loop_vars_.begin(), write_loop_vars_.end(), (*it)->name_hint) == write_loop_vars_.end()) {
                    write_loop_vars_.push_back((*it)->name_hint);
                    LOG(INFO) << "Loop variable used in write block: " << (*it)->name_hint;
                }
            }
        }
        // Mark buffers written in this reduction block
        for (const auto& write_region : block->writes) {
            const Buffer& buffer = write_region->buffer;
            reduction_buffers_.insert(buffer);
            LOG(INFO) << "Buffer written in reduction block: " << buffer->name;
        }
    } else {
        // Check if buffers in reduction_buffers_ are read in this block
        for (const auto& read_region : block->reads) {
            const Buffer& buffer = read_region->buffer;
            if (reduction_buffers_.count(buffer)) {
                LOG(INFO) << "Buffer read after reduction: " << buffer->name;
                is_read_after_reduction_ = true;
                std::unordered_set<std::string> used_indices;
                for (const auto& range : read_region->region) {
                  tvm::tir::PostOrderVisit(range->min, [&](const tvm::runtime::ObjectRef& expr) {
                  if (const VarNode* var_node = expr.as<VarNode>()) {
                      LOG(INFO) << "Found variable in min: " << var_node->name_hint;
                      used_indices.insert(var_node->name_hint);
                    }
                  });

                  tvm::tir::PostOrderVisit(range->extent, [&](const tvm::runtime::ObjectRef& expr) {
                      if (const VarNode* var_node = expr.as<VarNode>()) {
                          LOG(INFO) << "Found variable in extent: " << var_node->name_hint;
                          used_indices.insert(var_node->name_hint);
                      }
                  });
                }
                // Read Block에서 사용하는 루프 변수 저장
                if (!loop_vars_.empty()) {  // 빈 경우 예외 방지
                    for (auto it = loop_vars_.begin(); it != loop_vars_.end(); ++it) {
                        // 없는 경우만 추가 (std::find 사용)
                        if (std::find(read_loop_vars_.begin(), read_loop_vars_.end(), (*it)->name_hint) == read_loop_vars_.end()) {
                            if (!used_indices.count((*it)->name_hint)) {
                              read_loop_vars_.push_back((*it)->name_hint);
                            }
                            LOG(INFO) << "Loop variable used in read block: " << (*it)->name_hint;
                        }
                    }
                }

            }
        }
    }

    // Recursively visit the body
    StmtExprVisitor::VisitStmt_(op);
  }


  void VisitStmt_(const ForNode* op) final {
    LOG(INFO) << "Entering For loop: " << op->loop_var->name_hint
              << " at address: " << op->loop_var.get();

    // Push the loop_var onto the stack
    loop_vars_.push_back(op->loop_var);

    StmtExprVisitor::VisitStmt_(op);
    // Pop the loop_var from the stack
    loop_vars_.pop_back();
  }

  // Member variables
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> reduction_buffers_;
  bool is_read_after_reduction_{false};
  
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> reduction_loop_vars_;

  std::vector<std::string> write_loop_vars_;
  std::vector<std::string> read_loop_vars_;
  std::vector<std::string> shared_loop_vars_;
  std::vector<Var> loop_vars_;
};


PrimFunc ExpandReductionBuffers(PrimFunc func) {
  CheckReductionReadUsage checker;
  checker(func->body);

  LOG(INFO) << "Checking for read after reduction: " << (checker.IsReadAfterReduction() ? "true" : "false");
  if (checker.IsReadAfterReduction()) {
    checker.FindSharedLoopVars();  
    std::vector<std::string> shared_loop_vars = checker.GetSharedLoopVars();  // 3. 공유 루프 변수 가져오기
    LOG(INFO) << "Shared loop variables found: ";
    for (const auto& var : shared_loop_vars) {
        LOG(INFO) << var;
    }
    if (!shared_loop_vars.empty()) {
      ExpandReductionBuffersMutator expander(func, shared_loop_vars);
      auto* n = func.CopyOnWrite();
      n->body = expander(func->body);
    }
  }
  return func;
}



namespace transform {

Pass ExpandReductionBuffers() {
  auto pass_func = [](PrimFunc func, IRModule mod, PassContext ctx) {
    return ExpandReductionBuffers(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.ExpandReductionBuffers", {});
}

TVM_REGISTER_GLOBAL("tir.transform.ExpandReductionBuffers")
    .set_body_typed(ExpandReductionBuffers);

}  // namespace transform
}  // namespace tir
}  // namespace tvm