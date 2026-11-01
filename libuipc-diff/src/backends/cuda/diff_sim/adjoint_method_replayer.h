#pragma once
#include <sim_system.h>
#include <diff_sim/global_adjoint_method_replayer.h>
namespace uipc::backend::cuda
{
class AdjointMethodReplayer : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class BuildInfo
    {
      public:
    };

    using RecordInfo = GlobalAdjointMethodReplayer::RecordInfo;
    using ReplayInfo = GlobalAdjointMethodReplayer::ReplayInfo;

  protected:
    virtual void do_record(RecordInfo& info) = 0;
    virtual void do_replay(ReplayInfo& info) = 0;
    virtual void do_clear()                  = 0;
    virtual void do_build(BuildInfo& info)   = 0;

  private:
    friend class GlobalAdjointMethodReplayer;
    // only be called by GlobalAdjointMethodReplayer
    void record(RecordInfo& info);
    // only be called by GlobalAdjointMethodReplayer
    void replay(ReplayInfo& info);
    // only be called by GlobalAdjointMethodReplayer
    void clear();

    virtual void do_build() final override;
};
}  // namespace uipc::backend::cuda
