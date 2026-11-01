#include <coupling_system/abd_fem_contact_diff_dof_reporter.h>
#include <utils/matrix_unpacker.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(ABDFEMDiffDofReporter);

void ABDFEMDiffDofReporter::do_build(BuildInfo& info)
{
    m_impl.abd = &require<AffineBodyDynamics>();
    m_impl.fem = &require<FiniteElementMethod>();

    m_impl.abd_fem_receiver = &require<ABDFEMContactDiffDofReceiver>();
    m_impl.fem_abd_receiver = &require<FEMABDContactDiffDofReceiver>();

    m_impl.abd_vertex_reporter = &require<AffineBodyVertexReporter>();
    m_impl.fem_vertex_reporter = &require<FiniteElementVertexReporter>();
}

void ABDFEMDiffDofReporter::do_report_extent(DiffDofExtentInfo& info)
{
    auto abd_fem_H = m_impl.abd_fem_receiver->H();
    auto fem_abd_H = m_impl.fem_abd_receiver->H();

    auto F = info.frame();

    if(F >= 2)
    {
        // H12x3 = J^T * H3x3
        auto AF_triplet_count = abd_fem_H.triplet_count() * 4;
        // H3x12 = H3x3 * J
        auto FA_triplet_count = fem_abd_H.triplet_count() * 4;

        UIPC_ASSERT(AF_triplet_count == FA_triplet_count,
                    "ABD-FEM and FEM-ABD contact diff dof count mismatch, AF={}, FA={}",
                    AF_triplet_count,
                    FA_triplet_count);

        info.triplet_count(AF_triplet_count + FA_triplet_count);
    }
    else
    {
        info.triplet_count(0);
    }
}

void ABDFEMDiffDofReporter::do_assemble(DiffDofInfo& info)
{
    m_impl.assemble(info);
}

void ABDFEMDiffDofReporter::Impl::assemble(DiffDofInfo& info)
{
    auto F = info.frame();
    if(F < 2)
        return;

    auto F_dof_offset  = info.dof_offset(F);
    auto F_dof_count   = info.dof_count(F);
    auto F1_dof_offset = info.dof_offset(F - 1);
    auto F1_dof_count  = info.dof_count(F - 1);

    auto F_abd_dof_offset = F_dof_offset + abd->dof_offset(F);
    auto F_abd_dof_count  = abd->dof_count(F);

    auto F1_abd_dof_offset = F1_dof_offset + abd->dof_offset(F - 1);
    auto F1_abd_dof_count  = abd->dof_count(F - 1);

    auto F_fem_dof_offset = F_dof_offset + fem->dof_offset(F);
    auto F_fem_dof_count  = fem->dof_count(F);

    auto F1_fem_dof_offset = F1_dof_offset + fem->dof_offset(F - 1);
    auto F1_fem_dof_count  = fem->dof_count(F - 1);

    auto abd_global_vertex_offset = abd_vertex_reporter->vertex_offset();
    auto fem_global_vertex_offset = fem_vertex_reporter->vertex_offset();

    UIPC_ASSERT(F_dof_count == F_abd_dof_count + F_fem_dof_count,
                "ABD-FEM contact diff dof count mismatch, F={}, ABD={}, FEM={}, total={}",
                F,
                F_abd_dof_count,
                F_fem_dof_count,
                F_dof_count);

    UIPC_ASSERT(F1_dof_count == F1_abd_dof_count + F1_fem_dof_count,
                "ABD-FEM contact diff dof count mismatch, F={}, ABD={}, FEM={}, total={}",
                F - 1,
                F1_abd_dof_count,
                F1_fem_dof_count,
                F1_dof_count);

    UIPC_ASSERT(F_dof_count == F1_dof_count, "Now only support same dof count between F and F-1");

    using namespace muda;

    auto half_triplet_count = info.H().triplet_count() / 2;

    // abd-fem
    {
        auto pGpXn = abd_fem_receiver->H();


        // setup the range of the submatrix
        auto H = info.H()
                     .subview(0, half_triplet_count)
                     .submatrix({F_abd_dof_offset / 3, F1_fem_dof_offset / 3},
                                {F_abd_dof_count / 3, F1_fem_dof_count / 3});

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(pGpXn.triplet_count(),
                   [pGpXns = pGpXn.viewer().name("pGpXn"),
                    v2b    = abd->v2b().cviewer().name("v2b"),
                    Js     = abd->Js().viewer().name("Js"),
                    abd_is_fixed = abd->body_is_fixed().viewer().name("abd_is_fixed"),
                    fem_is_fixed = fem->is_fixed().viewer().name("fem_is_fixed"),
                    abd_global_vertex_offset,
                    fem_global_vertex_offset,
                    Hs = H.viewer().name("H")] __device__(int I) mutable
                   {
                       auto&& [global_i, global_j, pGpXn] = pGpXns(I);
                       TripletMatrixUnpacker TMU{Hs};

                       auto i = global_i - abd_global_vertex_offset;
                       auto j = global_j - fem_global_vertex_offset;

                       const ABDJacobi& Ji = Js(i);

                       Matrix12x3 H12x3;

                       auto bodyI = v2b(i);
                       auto vertJ = j;

                       if(abd_is_fixed(bodyI) || fem_is_fixed(vertJ))
                       {
                           H12x3.setZero();
                       }
                       else
                       {
                           H12x3 = Ji.to_mat().transpose() * pGpXn;
                       }

                       TMU.block<4, 1>(I * 4).write(bodyI * 4,  // bodyI
                                                    vertJ,      // vertJ
                                                    H12x3);
                   });
    }

    // fem-abd
    {
        auto pGpXn = fem_abd_receiver->H();

        // setup the range of the submatrix
        auto H = info.H()
                     .subview(half_triplet_count, half_triplet_count)
                     .submatrix({F_fem_dof_offset / 3, F1_abd_dof_offset / 3},
                                {F_fem_dof_count / 3, F1_abd_dof_count / 3});

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(pGpXn.triplet_count(),
                   [pGpXns = pGpXn.viewer().name("pGpXn"),
                    v2b    = abd->v2b().cviewer().name("v2b"),
                    Js     = abd->Js().viewer().name("Js"),
                    fem_is_fixed = fem->is_fixed().viewer().name("fem_is_fixed"),
                    abd_is_fixed = abd->body_is_fixed().viewer().name("abd_is_fixed"),
                    abd_global_vertex_offset,
                    fem_global_vertex_offset,
                    Hs = H.viewer().name("H")] __device__(int I) mutable
                   {
                       auto&& [global_i, global_j, pGpXn] = pGpXns(I);
                       TripletMatrixUnpacker TMU{Hs};

                       auto i = global_i - fem_global_vertex_offset;
                       auto j = global_j - abd_global_vertex_offset;

                       const ABDJacobi& Jj = Js(j);

                       Matrix3x12 H3x12;

                       auto vertI = i;
                       auto bodyJ = v2b(j);

                       if(fem_is_fixed(vertI) || abd_is_fixed(bodyJ))
                       {
                           H3x12.setZero();
                       }
                       else
                       {
                           H3x12 = pGpXn * Jj.to_mat();
                       }

                       TMU.block<1, 4>(I * 4).write(vertI,      // vertI
                                                    bodyJ * 4,  // bodyJ
                                                    H3x12);
                   });
    }
}
}  // namespace uipc::backend::cuda
