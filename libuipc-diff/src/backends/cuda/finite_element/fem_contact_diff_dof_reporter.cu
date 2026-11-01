#include <finite_element/fem_contact_diff_dof_reporter.h>
#include <utils/matrix_unpacker.h>
#include <kernel_cout.h>
namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(FEMContactDiffDofReporter);


void FEMContactDiffDofReporter::do_build(BuildInfo& info)
{
    m_impl.vertex_reporter = &require<FiniteElementVertexReporter>();
    m_impl.receiver        = &require<FEMContactDiffDofReceiver>();
    m_impl.fem             = &require<FiniteElementMethod>();
}

void FEMContactDiffDofReporter::do_report_extent(GlobalDiffSimManager::DiffDofExtentInfo& info)
{
    auto F = info.frame();

    if(F >= 2)
    {
        auto triplet_count = m_impl.receiver->H().triplet_count();
        info.triplet_count(triplet_count);
    }
    else
    {
        info.triplet_count(0);
    }
}

void FEMContactDiffDofReporter::do_assemble(DiffDofInfo& info)
{
    m_impl.assemble(info);
}
void FEMContactDiffDofReporter::Impl::assemble(DiffDofInfo& info)
{
    using namespace muda;

    auto global_vertex_offset = vertex_reporter->vertex_offset();

    auto F = info.frame();

    if(F >= 2)
    {
        auto row_offset     = info.dof_offset(F) / 3;
        auto col_dof_offset = info.dof_offset(F - 1) / 3;

        auto row_count = info.dof_count(F) / 3;
        auto col_count = info.dof_count(F - 1) / 3;

        UIPC_ASSERT(row_count == col_count, "Now only support same dof count for X and Xn");

        auto pGpXn = receiver->H();
        auto H = info.H().submatrix({row_offset, col_dof_offset}, {row_count, col_count});

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(pGpXn.triplet_count(),
                   [pGpXns = pGpXn.viewer().name("pGpXn"),
                    global_vertex_offset,
                    is_fixed = fem->is_fixed().viewer().name("is_fixed"),
                    Hs       = H.viewer().name("H")] __device__(int I) mutable
                   {
                       auto&& [global_i, global_j, pGpXn] = pGpXns(I);

                       auto i = global_i - global_vertex_offset;
                       auto j = global_j - global_vertex_offset;

                       if(is_fixed(i) || is_fixed(j))
                       {
                           Hs(I).write(i, j, Matrix3x3::Zero());
                       }
                       else
                       {
                           //tex:
                           //$$
                           //\frac{\partial G}{\partial X^n} = \frac{\partial^2 E}{\partial X \partial X^n}
                           //$$
                           Hs(I).write(i, j, pGpXn);
                       }
                       // cout << pGpXn << "\n";
                   });
    }
}
}  // namespace uipc::backend::cuda