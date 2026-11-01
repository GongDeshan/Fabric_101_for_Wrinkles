#pragma once
#include <gradient_hessian_computer.h>
#include <affine_body/affine_body_dynamics.h>
#include <uipc/diff_sim/enable_grad_feature.h>
#include <affine_body/affine_body_animator.h>

namespace uipc::backend::cuda
{
class ABDDiffSimManager;
class ABDGradientHessianComputer final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl
    {
      public:
        SimSystemSlot<AffineBodyDynamics> affine_body_dynamics;
        GlobalDiffSimManager*             global_diff_sim_manager = nullptr;
        ABDDiffSimManager*                abd_diff_sim_manager    = nullptr;
        SimSystemSlot<AffineBodyAnimator> affine_body_animator;
        AffineBodyDynamics::Impl&         abd() noexcept;

        void compute_gradient_hessian(GradientHessianComputer::ComputeInfo& info);

        // only used for diff sim
        muda::DeviceBuffer<Vector12> shape_gradient;
    };

  private:
    virtual void do_build() override;

    // allow AffineBodyConstitutionDiffVolume to access shape_gradient
    friend class AffineBodyConstitutionDiffVolume;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
