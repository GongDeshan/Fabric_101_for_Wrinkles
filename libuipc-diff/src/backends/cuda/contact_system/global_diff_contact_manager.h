#pragma once
#include <sim_system.h>
#include <muda/ext/linear_system.h>
#include <utils/offset_count_collection.h>
#include <algorithm/matrix_converter.h>
#include <global_geometry/global_vertex_manager.h>
#include <contact_system/contact_classify_info.h>
#include <contact_system/contact_coeff.h>
#include <diff_sim/global_diff_sim_manager.h>

namespace uipc::backend::cuda
{
class ContactDiffDofReporter;
class ContactDiffDofReceiver;
class ContactDiffParmReporter;
class ContactDiffParmReceiver;
class GlobalContactManager;

class GlobalDiffContactManager : public SimSystem
{
    //tex:
    // Now we assume the friction contact only related to
    // the position at current frame(k)
    // and the position at previous frame(k-1).
    // So we only need to store diff dof of:
    //
    //$$
    //H^{[k][k-1]} = \frac{\partial G^{[k]}}{\partial x^{[k-1]}}
    //$$

    //tex: The diff parm of the contact system is the
    //derivative of the contact gradient w.r.t. the parameters.
    //$$
    //pGpP = \frac{\partial G^{[k]}}{\partial P}
    //$$
  public:
    using SimSystem::SimSystem;

    class Impl;

    // Info for reporters:
    // 1) DiffDofReporter is responsible for computing the diff dof of the contact system.
    // 2) DiffParmReporter is responsible for computing the diff parm of the contact system.
    // 3) The manager will collect the diff dof and diff parm from all reporters and make them unique to BCOOMatrix.

    class ContactDiffDofExtentInfo
    {
      public:
        /*
        * @breif Set the count of the triplets in the diff dof matrix [Matrix3x3]
        * 
        * @param count The count of the triplets in the diff dof matrix [Matrix3x3]
        */
        void triplet_count(SizeT count) noexcept { m_triplet_count = count; }

      private:
        friend class Impl;
        SizeT m_triplet_count = 0;
    };

    class ContactDiffDofInfo
    {
      public:
        /*
        * @breif A subrange of the DiffDof Matrix [Matrix3x3]
        */
        auto H() const noexcept { return m_H; }

      private:
        friend class Impl;
        muda::TripletMatrixView<Float, 3> m_H;
    };

    class ContactDiffParmExtentInfo
    {
      public:
        /*
        * @breif Set the count of the triplets in the diff parm matrix [Matrix3x1]
        */
        void triplet_count(SizeT count) noexcept { m_triplet_count = count; }

      private:
        friend class Impl;
        SizeT m_triplet_count = 0;
    };

    class ContactDiffParmInfo
    {
      public:
        ContactDiffParmInfo(Impl* impl, muda::TripletMatrixView<Float, 3, 1> pGpP) noexcept
            : m_impl{impl}
            , m_pGpP{pGpP}
        {
        }

        /*
        * @breif A subrange of the DiffParm Matrix [Matrix3x1]
        */
        auto pGpP() const noexcept { return m_pGpP; }

      private:
        friend class Impl;
        Impl*                                m_impl;
        muda::TripletMatrixView<Float, 3, 1> m_pGpP;
    };

    // Info for receivers:
    // 1) The manager will classify the diff dof and diff parm according to the contact classify info (from the receiver).
    // 2) The manager will distribute the classified diff dof and diff parm to the receivers respectively.

    using ClassifyInfo = ContactClassifyInfo;

    class ClassifiedContactDiffDofInfo
    {
      public:
        auto H() const noexcept { return m_H; }

      private:
        friend class Impl;
        muda::CBCOOMatrixView<Float, 3> m_H;
    };

    class ClassifiedContactDiffParmInfo
    {
      public:
        auto pGpP() const noexcept { return m_pGpP; }

      private:
        friend class Impl;
        muda::CBCOOMatrixView<Float, 3, 1> m_pGpP;
    };

    class Impl
    {
      public:
        void init(WorldVisitor& world);
        void _build_diff_contact_coeff(WorldVisitor& world);
        void assemble_diff_dof();
        void assemble_diff_parm();
        void convert_matrix();
        void distribute_diff_dof();
        void distribute_diff_parm();
        void clear_contact_for_adjoint();

        GlobalVertexManager*  global_vertex_manager   = nullptr;
        GlobalContactManager* global_contact_manager  = nullptr;
        GlobalDiffSimManager* global_diff_sim_manager = nullptr;

        vector<DiffContactCoeff>               h_diff_contact_coeff;
        muda::DeviceBuffer2D<DiffContactCoeff> diff_contact_coeff;

        /***************************************************************************
        *                               Diff Dof 
        ***************************************************************************/

        SimSystemSlotCollection<ContactDiffDofReporter> diff_dof_reporters;
        SimSystemSlotCollection<ContactDiffDofReceiver> diff_dof_receivers;

        OffsetCountCollection<SizeT> diff_dof_reporter_offsets_counts;

        // assembled diff dof
        muda::DeviceTripletMatrix<Float, 3> triplet_H;
        // sorted and unique diff dof
        muda::DeviceBCOOMatrix<Float, 3> bcoo_H;
        MatrixConverter<Float, 3>        matrix_converter_H;

        muda::DeviceBuffer<IndexT>                  selected_hessian;
        muda::DeviceBuffer<IndexT>                  selected_hessian_offsets;
        vector<muda::DeviceTripletMatrix<Float, 3>> classified_H;

        /***************************************************************************
        *                               Diff Parm
        *****************************************************************************/

        SimSystemSlotCollection<ContactDiffParmReporter> diff_parm_reporters;
        SimSystemSlotCollection<ContactDiffParmReceiver> diff_parm_receivers;

        OffsetCountCollection<SizeT> diff_parm_reporter_offsets_counts;

        // assembled diff parm
        muda::DeviceTripletMatrix<Float, 3, 1> triplet_pGpP;
        // sorted and unique diff parm
        muda::DeviceBCOOMatrix<Float, 3, 1> bcoo_pGpP;
        MatrixConverter<Float, 3, 1>        matrix_converter_pGpP;

        muda::DeviceBuffer<IndexT>                     selected_pGpP;
        muda::DeviceBuffer<IndexT>                     selected_pGpP_offsets;
        vector<muda::DeviceTripletMatrix<Float, 3, 1>> classified_pGpP;


        Float reserve_ratio = 1.1;
        template <typename T>
        void loose_resize(muda::DeviceBuffer<T>& buffer, SizeT size)
        {
            if(size > buffer.capacity())
            {
                buffer.reserve(size * reserve_ratio);
            }
            buffer.resize(size);
        }

        template <int M, int N>
        void loose_resize_entries(muda::DeviceTripletMatrix<Float, M, N>& m, SizeT size)
        {
            if(size > m.triplet_capacity())
            {
                m.reserve_triplets(size * reserve_ratio);
            }
            m.resize_triplets(size);
        }
    };

    auto diff_contact_tabular() const noexcept
    {
        return m_impl.diff_contact_coeff.view();
    }


  private:
    virtual void do_build() final override;

    friend class ContactDiffDofReporter;
    friend class ContactDiffDofReceiver;
    friend class ContactDiffParmReporter;
    friend class ContactDiffParmReceiver;

    void add_reporter(ContactDiffDofReporter* reporter) noexcept;
    void add_receiver(ContactDiffDofReceiver* receiver) noexcept;
    void add_reporter(ContactDiffParmReporter* reporter) noexcept;
    void add_receiver(ContactDiffParmReceiver* receiver) noexcept;

    friend class SimEngine;
    void init();  // only be called by SimEngine

    friend class GlobalDiffSimManager;
    void update();

    friend class AdjointMethod;
    void compute_diff_contact();  // only be called by AdjointMethod
    void clear_contact_for_adjoint();  // Clear both history and parameter derivatives.

    Impl m_impl;
};
}  // namespace uipc::backend::cuda