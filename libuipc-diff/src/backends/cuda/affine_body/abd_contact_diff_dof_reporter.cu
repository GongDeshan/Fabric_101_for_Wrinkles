#include <affine_body/abd_contact_diff_dof_reporter.h>
#include <utils/matrix_unpacker.h>
#include <kernel_cout.h>
namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(ABDContactDiffDofReporter);


void ABDContactDiffDofReporter::do_build(BuildInfo& info)
{
    m_impl.abd             = &require<AffineBodyDynamics>();
    m_impl.vertex_reporter = &require<AffineBodyVertexReporter>();
    m_impl.receiver        = &require<ABDContactDiffDofReceiver>();
}

void ABDContactDiffDofReporter::do_report_extent(GlobalDiffSimManager::DiffDofExtentInfo& info)
{
    auto F = info.frame();

    if(F >= 2)
    {
        // contact hessian matrix is H3x3 of vertex-pair (i,j)
        auto triplet_count = m_impl.receiver->H().triplet_count();

        // affine body contact hessian matrix is H12x12 = J_i^T * H3x3 * J_j
        info.triplet_count(triplet_count * 4 * 4);
    }
    else
    {
        info.triplet_count(0);
    }
}

void ABDContactDiffDofReporter::do_assemble(DiffDofInfo& info)
{
    m_impl.assemble(info);
}

void ABDContactDiffDofReporter::Impl::assemble(DiffDofInfo& info)
{
    using namespace muda;

    auto global_vertex_offset = vertex_reporter->vertex_offset();

    auto F = info.frame();

    if(F >= 2)
    {
        auto row_offset = info.dof_offset(F) / 3;
        auto col_offset = info.dof_offset(F - 1) / 3;

        auto row_count = info.dof_count(F) / 3;
        auto col_count = info.dof_count(F - 1) / 3;

        UIPC_ASSERT(row_count == col_count, "Now only support same dof count for X and Xn");

        auto pGpXn = receiver->H();
        auto H = info.H().submatrix({row_offset, col_offset}, {row_count, col_count});

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(pGpXn.triplet_count(),
                   [pGpXns = pGpXn.viewer().name("pGpXn"),
                    v2b    = abd->v2b().cviewer().name("v2b"),
                    Js     = abd->Js().viewer().name("Js"),
                    body_is_fixed = abd->body_is_fixed().viewer().name("body_is_fixed"),
                    global_vertex_offset,
                    Hs = H.viewer().name("H")] __device__(int I) mutable
                   {
                       auto&& [global_i, global_j, pGpXn] = pGpXns(I);
                       TripletMatrixUnpacker TMU{Hs};

                       auto i = global_i - global_vertex_offset;
                       auto j = global_j - global_vertex_offset;

                       const ABDJacobi& Ji = Js(i);
                       const ABDJacobi& Jj = Js(j);

                       Matrix12x12 H12;

                       auto bodyI = v2b(i);
                       auto bodyJ = v2b(j);

                       if(body_is_fixed(bodyI) || body_is_fixed(bodyJ))
                       {
                           H12.setZero();
                       }
                       else
                       {
                           H12 = ABDJacobi::JT_H_J(Ji.T(), pGpXn, Jj);
                       }


                       TMU.block<4, 4>(I * 4 * 4)  // 16 * matrix3x3
                           .write(bodyI * 4,       // bodyI
                                  bodyJ * 4,       // bodyJ
                                  H12);
                   });
    }
}
}  // namespace uipc::backend::cuda