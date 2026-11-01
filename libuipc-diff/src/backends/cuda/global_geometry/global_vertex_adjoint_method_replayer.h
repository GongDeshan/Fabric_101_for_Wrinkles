#pragma once
#include <diff_sim/adjoint_method_replayer.h>
#include <muda/buffer.h>
#include <stack>

namespace uipc::backend::cuda
{
class GlobalVertexManager;
class GlobalVertexAdjointMethodReplayer final : public AdjointMethodReplayer
{
  public:
    using AdjointMethodReplayer::AdjointMethodReplayer;
    class Impl
    {
      public:
        GlobalVertexManager*        global_vertex_manager = nullptr;
        muda::DeviceBuffer<Vector3> positions;
        muda::DeviceBuffer<Vector3> prev_positions;
        std::stack<IndexT>          vertex_offset_stack;
        std::stack<IndexT>          vertex_count_stack;
    };

  private:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_record(RecordInfo& info) override;
    virtual void do_replay(ReplayInfo& info) override;
    virtual void do_clear() override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
