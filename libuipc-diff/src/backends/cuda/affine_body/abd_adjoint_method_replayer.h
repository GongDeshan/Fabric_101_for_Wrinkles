#pragma once
#include <diff_sim/adjoint_method_replayer.h>
#include <muda/buffer.h>
#include <stack>

namespace uipc::backend::cuda
{
class AffineBodyDynamics;
class ABDAdjointMethodReplayer final : public AdjointMethodReplayer
{
  public:
    using AdjointMethodReplayer::AdjointMethodReplayer;
    class Impl
    {
      public:
        AffineBodyDynamics* affine_body_dynamics = nullptr;

        // forward simulation requestion
        muda::DeviceBuffer<Vector12> qs;
        muda::DeviceBuffer<Vector12> q_tildes;

        // abd kinetic diff rest shape need:
        // - q_vs
        // - q_prevs
        muda::DeviceBuffer<Vector12> q_vs;
        muda::DeviceBuffer<Vector12> q_prevs;

        std::stack<IndexT> body_offset_stack;
    };

  private:
    virtual void do_build(BuildInfo& info) override;

    virtual void do_record(RecordInfo& info) override;
    virtual void do_replay(ReplayInfo& info) override;
    virtual void do_clear() override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
