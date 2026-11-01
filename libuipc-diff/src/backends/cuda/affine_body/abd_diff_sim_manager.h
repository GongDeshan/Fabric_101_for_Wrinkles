#pragma once
#include <sim_system.h>
#include <muda/buffer/device_buffer.h>
#include <diff_sim/global_diff_sim_manager.h>
#include <affine_body/abd_jacobi_matrix.h>

namespace uipc::backend::cuda
{
class AffineBodyDynamics;
class GlobalDiffSimManager;

class ABDDiffSimManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl
    {
      public:
        void init(backend::WorldVisitor& world);
        void update_parms(GlobalDiffSimManager::DiffParmUpdateInfo& info);

        //vector<Vector3i>             h_triangles;
        //muda::DeviceBuffer<Vector3i> triangles;

        // triangles that need diff optimization
        vector<Vector3i>             h_diff_triangles;
        muda::DeviceBuffer<Vector3i> diff_triangles;

        // tetrahedra that need diff optimization
        vector<Vector4i>             h_diff_tetrahedra;
        muda::DeviceBuffer<Vector4i> diff_tetrahedra;
        // vertices that need diff optimization
        vector<IndexT>             h_diff_vertices;
        muda::DeviceBuffer<IndexT> diff_vertices;
        // bodies that need diff optimization
        vector<IndexT>             h_diff_bodies;
        muda::DeviceBuffer<IndexT> diff_bodies;

        vector<Float>             h_body_rho;
        muda::DeviceBuffer<Float> body_rho;

        // a mapping from vertex to DiffSim parameters
        // -1: this vertex is not a DiffSim parameter
        // >=0: the index of the DiffSim parameter
        vector<Vector3i>             h_diff_rest_positions;
        muda::DeviceBuffer<Vector3i> diff_rest_positions;

        vector<Vector3>             h_instance_gravity;
        muda::DeviceBuffer<Vector3> instance_gravity;


        GlobalDiffSimManager* global_diff_sim_manager = nullptr;
        AffineBodyDynamics*   affine_body_dynamics    = nullptr;
    };

    muda::CBufferView<Vector3i> diff_triangles() const noexcept
    {
        return m_impl.diff_triangles;
    }

    muda::CBufferView<Vector4i> diff_tetrahedra() const noexcept
    {
        return m_impl.diff_tetrahedra;
    }

    muda::CBufferView<Vector3i> diff_rest_positions() const noexcept
    {
        return m_impl.diff_rest_positions;
    }

    muda::CBufferView<IndexT> diff_vertices() const noexcept
    {
        return m_impl.diff_vertices;
    }

    muda::CBufferView<IndexT> diff_bodies() const noexcept
    {
        return m_impl.diff_bodies;
    }

    muda::CBufferView<Float> body_mass_density() const noexcept
    {
        return m_impl.body_rho;
    }

    muda::CBufferView<Vector3> instance_gravity() const noexcept
    {
        return m_impl.instance_gravity;
    }

  private:
    friend class SimEngine;

    void init();  // only be called by SimEngine


    // Internal:
    virtual void do_build() override;

    Impl m_impl;
};
}  // namespace uipc::backend::cuda
