#pragma once
#include <sim_system.h>
#include <muda/ext/linear_system.h>
#include <utils/offset_count_collection.h>
#include <algorithm/matrix_converter.h>

namespace uipc::diff_sim
{
class EnableGradFeature;
}

namespace uipc::backend::cuda
{
class DiffDofReporter;
class DiffParmReporter;
class GlobalLinearSystem;
class GlobalAdjointMethodReplayer;
class GlobalVertexManager;
class GlobalDiffContactManager;

class GlobalDiffSimManager final : public SimSystem
{
  public:
    using SimSystem::SimSystem;

    class Impl;

    class BaseInfo
    {
      public:
        BaseInfo(Impl* impl, SizeT index)
            : m_impl(impl)
            , m_index(index)
        {
        }

        SizeT  frame() const;
        IndexT dof_offset(SizeT frame) const;
        IndexT dof_count(SizeT frame) const;

      protected:
        friend class Impl;
        Impl* m_impl  = nullptr;
        SizeT m_index = ~0ull;
    };

    class DiffParmExtentInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;
        void triplet_count(SizeT N) { m_triplet_count = N; }

      private:
        friend class Impl;
        SizeT m_triplet_count = 0;
    };


    class DiffParmInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;
        muda::TripletMatrixView<Float, 3, 1> pGpP() const;
    };

    class DiffDofExtentInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;
        void triplet_count(SizeT N) { m_triplet_count = N; }

      private:
        friend class Impl;
        SizeT m_triplet_count = 0;
    };


    class DiffDofInfo : public BaseInfo
    {
      public:
        using BaseInfo::BaseInfo;

        friend class Impl;
        muda::TripletMatrixView<Float, 3, 3> H() const;
    };

    class DiffParmUpdateInfo
    {
      public:
        DiffParmUpdateInfo(Impl* impl)
            : m_impl(impl)
        {
        }

        muda::CBufferView<Float> parameters() const noexcept;


      private:
        friend class Impl;
        Impl* m_impl = nullptr;
    };

    class Impl
    {
      public:
        void init(WorldVisitor& world);
        void update(muda::CBufferView<Float> parms);
        void assemble(SizeT frame);
        void write_scene(WorldVisitor& world);

        GlobalLinearSystem*          global_linear_system           = nullptr;
        SimEngine*                   sim_engine                     = nullptr;
        GlobalAdjointMethodReplayer* global_adjoint_method_replayer = nullptr;
        GlobalVertexManager*         global_vertex_manager          = nullptr;
        GlobalDiffContactManager*    global_diff_contact_manager    = nullptr;

        SimSystemSlotCollection<DiffDofReporter>  diff_dof_reporters;
        SimSystemSlotCollection<DiffParmReporter> diff_parm_reporters;

        OffsetCountCollection<IndexT> diff_dof_triplet_offset_count;
        OffsetCountCollection<IndexT> diff_parm_triplet_offset_count;

        SizeT                                          total_parm_count = 0;
        muda::DeviceBuffer<Float>                      parameters;
        SimActionCollection<void(DiffParmUpdateInfo&)> on_parm_update;

        // NOTE:
        // local_triplet_pGpP only consider the triplet at current frame.
        // So the triplet count = current_frame_triplet_count
        // The shape of the matrix = (total_frame_dof_count, total_parm_count)
        //tex:
        //$$
        //T = \begin{bmatrix}
        // 0   \\
        // ... \\
        // 0   \\
        // T^{[i]} \\
        //\end{bmatrix}
        //$$
        muda::DeviceTripletMatrix<Float, 3, 1> local_triplet_pGpP;

        // NOTE:
        // local_triplet_H only consider the triplet at current frame.
        // So the triplet count = current_frame_triplet_count
        // The shape of the matrix = (total_frame_dof_count, total_frame_dof_count)
        //tex:
        //$$
        //T = \begin{bmatrix}
        // 0   \\
        // ... \\
        // 0   \\
        // H^{[i]} \\
        //\end{bmatrix}
        //$$
        muda::DeviceTripletMatrix<Float, 3, 3> local_triplet_H;

        S<diff_sim::EnableGradFeature> enable_grad_feature;
    };

    IndexT                   dof_offset(SizeT frame) const;
    IndexT                   dof_count(SizeT frame) const;
    muda::CBufferView<Float> parameters() const;

    muda::CTripletMatrixView<Float, 3, 1> pGipP() const;
    muda::CTripletMatrixView<Float, 3, 3> Hij() const;

    bool is_grad_enabled() const;
    void on_parm_update(SimSystem* system, std::function<void(DiffParmUpdateInfo&)>&& f);

  private:
    virtual void do_build() override;

    friend class SimEngine;
    void init();  // only be called by SimEngine

    friend class AdjointMethod;
    void assemble(SizeT frame);  // only be called by AdjointMethod

    friend class UpdateParameterFeatureOverrider;
    friend class GlobalLearningRateManager;
    void update(muda::CBufferView<Float> parms);  // only be called by UpdateParameterFeature and GlobalLearningRateManager

    friend class DiffDofReporter;
    friend class DiffParmReporter;
    void add_reporter(DiffDofReporter* subsystem);  // only be called by DiffDofReporter
    void add_reporter(DiffParmReporter* subsystem);  // only be called by DiffParmReporter

    Impl m_impl;
};
}  // namespace uipc::backend::cuda