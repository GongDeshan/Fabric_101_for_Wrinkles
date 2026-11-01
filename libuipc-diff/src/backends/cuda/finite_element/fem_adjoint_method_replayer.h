#pragma once
#include <diff_sim/adjoint_method_replayer.h>
#include <muda/buffer.h>
#include <stack>

namespace uipc::backend::cuda
{
class FiniteElementMethod;
class FEMAdjointMethodReplayer final : public AdjointMethodReplayer
{
  public:
    using AdjointMethodReplayer::AdjointMethodReplayer;
    class Impl
    {
      public:
        FiniteElementMethod*        finite_element_method = nullptr;
        muda::DeviceBuffer<Vector3> xs;
        muda::DeviceBuffer<Vector3> x_tildes;
        std::stack<IndexT>          vertex_offset_stack;
        IndexT                      current_vertex_offset = 0;
    };

  private:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_record(RecordInfo& info) override;
    virtual void do_replay(ReplayInfo& info) override;
    virtual void do_clear() override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
