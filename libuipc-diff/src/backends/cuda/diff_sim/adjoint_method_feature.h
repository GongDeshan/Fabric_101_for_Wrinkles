#pragma once
#include <sim_system.h>
#include <uipc/diff_sim/adjoint_method_feature.h>
#include <muda/buffer.h>

namespace uipc::backend::cuda
{
class AdjointMethod;
class AdjointMethodFeatureOverrider final : public diff_sim::AdjointMethodFeatureOverrider
{
  public:
    AdjointMethodFeatureOverrider(AdjointMethod& am);

    class Impl
    {
      public:
        muda::DeviceBuffer<IndexT> local_SDI;
        muda::DeviceBuffer<Float>  dofs;
        muda::DeviceBuffer<Float>  dLdP;
        muda::DeviceBuffer<Float>  dLdX;
    };

  protected:
    virtual void do_select_dofs(SizeT frame, backend::BufferView in_SDI) override;
    virtual void do_receive_dofs(backend::BufferView out_dofs) override;
    virtual void do_compute_dLdP(backend::BufferView out_dLdP,
                                 backend::BufferView in_dLdx) override;

  private:
    AdjointMethod& m_am;
    Impl           m_impl;
};
}  // namespace uipc::backend::cuda
