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
#include "../utils.h"

namespace tvm {
namespace meta_schedule {

/*! \brief The union of design space generators. */
class ScheduleFnNode : public SpaceGeneratorNode {
 public:
  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;
  /*! \brief The schedule function. */
  runtime::PackedFunc schedule_fn_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    SpaceGeneratorNode::VisitAttrs(v);
    // `schedule_fn_` is not visited.
  }

  void InitializeWithTuneContext(const TuneContext& context) final {
    SpaceGeneratorNode::InitializeWithTuneContext(context);
    this->rand_state_ = ForkSeed(&context->rand_state);
  }

  Array<tir::Schedule> GenerateDesignSpace(const IRModule& mod) final {
    tir::Schedule sch = tir::Schedule::Traced(
        /*mod=*/mod,
        /*rand_state=*/ForkSeed(&this->rand_state_),
        /*debug_mode=*/0,
        /*error_render_level=*/tir::ScheduleErrorRenderLevel::kDetail);
    runtime::TVMRetValue rv;
    rv = this->schedule_fn_(sch);
    if (rv.type_code() == kTVMNullptr) {
      return {sch};
    }
    ObjectRef obj = rv;
    if (auto sch = obj.as<tir::Schedule>()) {
      return GenerateDesignSpace_(mod, sch.value());
      // return {sch.value()};
    }
    if (const auto* arr = obj.as<runtime::ArrayNode>()) {
      Array<tir::Schedule> result;
      result.reserve(arr->size());
      for (const ObjectRef& obj : *arr) {
        if (auto sch = obj.as<tir::Schedule>()) {
          auto ss = GenerateDesignSpace_(mod, sch.value());
          for (auto s : ss) {
            result.push_back(s);
          }
        } else {
          LOG(FATAL) << "TypeError: Expect return type of ScheduleFn to be None, Schedule or "
                        "List[Schedule], but got: "
                     << obj->GetTypeKey();
        }
      }
      return result;
    }
    LOG(FATAL) << "TypeError: Expect return type of ScheduleFn to be None, Schedule or "
                  "List[Schedule], but got: "
               << obj->GetTypeKey();
    throw;
  }

  Array<tir::Schedule> GenerateDesignSpace_(const IRModule& mod, const tir::Schedule& sch) {
    using ScheduleAndUnvisitedBlocks = std::pair<tir::Schedule, Array<tir::BlockRV>>;
    CHECK(sch_rules.defined()) << "ValueError: `sch_rules` is not set in PostOrderApply";

    std::vector<ScheduleAndUnvisitedBlocks> stack;
    Array<tir::Schedule> result{sch};
    Array<tir::BlockRV> all_blocks = BlockCollector::Collect(sch, nullptr);

    for (ScheduleRule sch_rule : sch_rules.value()) {
      for (const tir::Schedule& sch : result) {
        stack.emplace_back(sch, all_blocks);
      }
      result.clear();
      while (!stack.empty()) {
        // get the stack.top()
        auto [sch, blocks] = stack.back();
        stack.pop_back();
        // if all blocks are visited
        if (blocks.empty()) {
          result.push_back(sch);
          continue;
        }
        // otherwise, get the last block that is not visited
        tir::BlockRV block_rv = blocks.back();
        blocks.pop_back();
        if (!sch->HasBlock(block_rv)) {
          stack.emplace_back(sch, blocks);
          continue;
        }
        if (!ScheduleRule::IsApplyCustomRule(sch_rule)) {
          if (tir::GetAnn<String>(sch->GetSRef(block_rv), "schedule_rule").defined()) {
            stack.emplace_back(sch, blocks);
            continue;
          }
        }

        Array<tir::Schedule> applied = sch_rule->Apply(sch, /*block=*/block_rv);
        for (const tir::Schedule& sch : applied) {
          stack.emplace_back(sch, blocks);
        }
      }
    }
    return result;
  }

  SpaceGenerator Clone() const final {
    ObjectPtr<ScheduleFnNode> n = make_object<ScheduleFnNode>(*this);
    CloneRules(this, n.get());
    return SpaceGenerator(n);
  }

  static constexpr const char* _type_key = "meta_schedule.ScheduleFn";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleFnNode, SpaceGeneratorNode);
};

SpaceGenerator SpaceGenerator::ScheduleFn(PackedFunc schedule_fn,
                                          Optional<Array<ScheduleRule>> sch_rules,
                                          Optional<Array<Postproc>> postprocs,
                                          Optional<Map<Mutator, FloatImm>> mutator_probs) {
  ObjectPtr<ScheduleFnNode> n = make_object<ScheduleFnNode>();
  n->sch_rules = std::move(sch_rules);
  n->postprocs = std::move(postprocs);
  n->mutator_probs = std::move(mutator_probs);
  n->schedule_fn_ = std::move(schedule_fn);
  return SpaceGenerator(n);
}

TVM_REGISTER_NODE_TYPE(ScheduleFnNode);
TVM_REGISTER_GLOBAL("meta_schedule.SpaceGeneratorScheduleFn")
    .set_body_typed(SpaceGenerator::ScheduleFn);

}  // namespace meta_schedule
}  // namespace tvm
