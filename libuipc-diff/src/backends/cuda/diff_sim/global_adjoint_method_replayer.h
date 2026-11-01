#pragma once
#include <sim_system.h>
#include <sim_system_slot.h>

namespace uipc::backend::cuda
{
class AdjointMethodReplayer;
class SimEngine;
class GlobalLinearSystem;

class GlobalAdjointMethodReplayer final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class ReplayInfo
    {
      public:
        auto frame() const { return m_frame; }

      private:
        friend class GlobalAdjointMethodReplayer;
        SizeT m_frame;
    };

    class RecordInfo
    {
      public:
        auto frame() const { return m_frame; }

      private:
        friend class GlobalAdjointMethodReplayer;
        SizeT m_frame;
    };

    class Impl
    {
      public:
        SimSystemSlotCollection<AdjointMethodReplayer> replayers;
        GlobalLinearSystem* global_linear_system = nullptr;

        vector<IndexT> frame_dof_offsets;
        vector<IndexT> frame_dof_counts;
    };

    IndexT dof_offset(SizeT frame) const;
    IndexT dof_count(SizeT frame) const;

  private:
    virtual void do_build() override;

    friend class SimEngine;
    /**
     * @brief Trigger a stage to record the current state of the system.
     * 
     * only be called by SimEngine::do_advance()
     */
    void record(bool enable_grad, SizeT frame);

    friend class AdjointMethod;
    /**
     * @brief Trigger a stage to replay the recorded state of the system at given frame.
     * 
     * only be called by AdjointMethod
     */
    void replay(SizeT frame);

    friend class AdjointMethodReplayer;
    // only be called by AdjointMethodReplayer
    void add_replayer(AdjointMethodReplayer* replayer);

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
