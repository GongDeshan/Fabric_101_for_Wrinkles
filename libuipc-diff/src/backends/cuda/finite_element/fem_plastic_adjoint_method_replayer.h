#pragma once
#include <diff_sim/adjoint_method_replayer.h>
#include <muda/buffer.h>
#include <stack>

namespace uipc::backend::cuda
{
class FiniteElementMethod;
class FEMPlasticAdjointMethodReplayer final : public AdjointMethodReplayer
{
  public:
    using AdjointMethodReplayer::AdjointMethodReplayer;
    class Impl
    {
      public:
        FiniteElementMethod* finite_element_method = nullptr;

        muda::DeviceBuffer<Float> anchor_thetas;
        muda::DeviceBuffer<Float> plastic_thetas;
        muda::DeviceBuffer<Float> dihedral_thetas;
        std::stack<IndexT>        edge_offset_stack;
        IndexT                    current_edge_offset = 0;

        muda::DeviceBuffer<Float> anchor_strain_uus, anchor_strain_vvs, anchor_strain_uvs;
        muda::DeviceBuffer<Float> yield_strain_uus, yield_strain_vvs, yield_strain_uvs;
        muda::DeviceBuffer<Float> hardening_plastic_strain_uus,
            hardening_plastic_strain_vvs, hardening_plastic_strain_uvs;
        muda::DeviceBuffer<Float> plastic_strain_uus, plastic_strain_vvs, plastic_strain_uvs;
        muda::DeviceBuffer<Float> strain_uus, strain_vvs, strain_uvs;

        muda::DeviceBuffer<Float> d_anchor_uv_d_thres_uv;
        muda::DeviceBuffer<Float> d_plastic_uv_d_yield_ori,
            d_plastic_uv_d_hardening_k33, d_plastic_uv_d_k33;

        muda::DeviceBuffer<Float> d_anchor_theta_d_thres_theta;
        muda::DeviceBuffer<Float> d_plastic_theta_d_yield_ori,
            d_plastic_theta_d_k, d_plastic_theta_d_hardening_k;

        muda::DeviceBuffer<Float> anchor_thetas_warp, anchor_thetas_weft;
        muda::DeviceBuffer<Float> plastic_thetas_warp, plastic_thetas_weft;

        std::stack<IndexT> orthotropic_edge_offset_stack;


        muda::DeviceBuffer<Float> d_anchor_theta_d_thres_theta_warp,
            d_anchor_theta_d_thres_theta_weft;
        muda::DeviceBuffer<Float> d_plastic_theta_d_k_warp, d_plastic_theta_d_k_weft;
        muda::DeviceBuffer<Float> d_plastic_theta_d_yield_ori_warp,
            d_plastic_theta_d_yield_ori_weft;
        muda::DeviceBuffer<Float> d_plastic_theta_d_hardening_k_warp,
            d_plastic_theta_d_hardening_k_weft;

        std::stack<IndexT> face_offset_stack;
        IndexT             current_face_offset = 0;
    };

  private:
    virtual void do_build(BuildInfo& info) override;
    virtual void do_record(RecordInfo& info) override;
    virtual void do_replay(ReplayInfo& info) override;
    virtual void do_clear() override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
