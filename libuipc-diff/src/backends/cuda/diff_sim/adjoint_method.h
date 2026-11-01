#pragma once
#include <sim_system.h>
#include <muda/buffer/buffer_view.h>
#include <muda/ext/linear_system/device_dense_vector.h>

namespace uipc::backend::cuda
{
class GlobalLinearSystem;
class GlobalDiffSimManager;
class GlobalAdjointMethodReplayer;
class GlobalTrajectoryFilter;
class GlobalContactManager;
class GlobalDiffContactManager;
class GradientHessianComputer;

class AdjointMethod final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl
    {
      public:
        // H[ii] * lambda[i] = f[i]
        muda::DeviceDenseVector<Float> f;         // full f[0...N-1]
        muda::DeviceDenseVector<Float> lambda_i;  // lambda[i]
        muda::DeviceBuffer<IndexT>     SDI;       // Selected Dof Indices
        muda::DeviceBuffer<Float>      dofs;      // Selected Dofs

        GlobalLinearSystem*      global_linear_system      = nullptr;  // H[ii]
        GlobalDiffSimManager*    global_diff_sim_manager   = nullptr;
        GradientHessianComputer* gradient_hessian_computer = nullptr;

        GlobalAdjointMethodReplayer* global_adjoint_method_replayer = nullptr;
        GlobalTrajectoryFilter*      global_trajectory_filter       = nullptr;
        GlobalContactManager*        global_contact_manager         = nullptr;
        GlobalDiffContactManager*    global_diff_contact_manager    = nullptr;
        bool contact_gradient = true;  // Opt-in approximation; forward is unchanged.
    };

  private:
    virtual void do_build() override;

    friend class AdjointMethodFeatureOverrider;
    void select_dofs(SizeT frame, muda::CBufferView<IndexT> in_SDI);
    void receive_dofs(muda::BufferView<Float> out_dofs);
    // only be called by AdjointMethodFeatureOverrider
    void compute_dLdP(muda::BufferView<Float> out_dLdP, muda::CBufferView<Float> in_dLdx);
    void assemble(SizeT frame);


    Impl m_impl;
};
}  // namespace uipc::backend::cuda
