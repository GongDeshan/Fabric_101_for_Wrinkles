#pragma once
#include <sim_system.h>
#include <diff_sim/global_learning_rate_manager.h>

namespace uipc::backend::cuda
{
class GlobalLearningRateManager;
class LearningRateReporter : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    class BuildInfo
    {
      public:
    };

    using LearningRateInfo    = GlobalLearningRateManager::LearningRateInfo;
    using RecordSafeStateInfo = GlobalLearningRateManager::RecordSafeStateInfo;

  protected:
    virtual void do_build(BuildInfo& info) = 0;

    virtual void record_safe_state(RecordSafeStateInfo& info) = 0;
    virtual void report_learning_rate(LearningRateInfo& info) = 0;

  private:
    friend class GlobalLearningRateManager;
    virtual void do_build() override final;
};
}  // namespace uipc::backend::cuda
