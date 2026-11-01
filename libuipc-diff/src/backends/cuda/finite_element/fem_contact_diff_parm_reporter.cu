#include <finite_element/fem_contact_diff_parm_reporter.h>
#include <utils/matrix_unpacker.h>
#include <kernel_cout.h>
namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(FEMContactDiffParmReporter);


void FEMContactDiffParmReporter::do_build(BuildInfo& info)
{
    m_impl.vertex_reporter = &require<FiniteElementVertexReporter>();
    m_impl.receiver        = &require<FEMContactDiffParmReceiver>();
    m_impl.fem             = &require<FiniteElementMethod>();
}

void FEMContactDiffParmReporter::do_report_extent(GlobalDiffSimManager::DiffParmExtentInfo& info)
{
    auto triplet_count = m_impl.receiver->pGpP().triplet_count();
    info.triplet_count(triplet_count);  //  3x1 matrix
}

void FEMContactDiffParmReporter::do_assemble(DiffParmInfo& info)
{
    m_impl.assemble(info);
}

void FEMContactDiffParmReporter::do_update(GlobalDiffSimManager::DiffParmUpdateInfo& info)
{
}

void FEMContactDiffParmReporter::do_init() {}

void FEMContactDiffParmReporter::Impl::assemble(DiffParmInfo& info)
{
    using namespace muda;

    auto global_vertex_offset = vertex_reporter->vertex_offset();

    auto src_pGpP = receiver->pGpP();
    auto dst_pGpP = info.pGpP();

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(src_pGpP.triplet_count(),
               [src_pGpPs = src_pGpP.cviewer().name("src_pGpPs"),
                global_vertex_offset,
                is_fixed = fem->is_fixed().viewer().name("is_fixed"),
                dst_pGpPs = dst_pGpP.viewer().name("dst_pGpPs")] __device__(int I) mutable
               {
                   auto&& [global_i, parm_j, pGpP] = src_pGpPs(I);
                   TripletMatrixUnpacker TMU{dst_pGpPs};

                   auto vertI = global_i - global_vertex_offset;
                   auto parmJ = parm_j;

                   //tex:
                   //$$
                   //\frac{\partial G}{\partial P} = \frac{\partial^2 E}{\partial X \partial P}
                   //$$

                   Vector3 pGpPValue = pGpP;

                   // If vertex is fixed, the gradient `G(X)` is always zero
                   // so pGpP is zero as well.
                   if(is_fixed(vertI))
                   {
                       pGpPValue.setZero();
                   }

                   dst_pGpPs(I).write(vertI,     // vertI
                                      parmJ,     // parmJ
                                      pGpPValue  // pGpP
                   );

                   cout << "FEM pGpP Contact:" << pGpPValue.transpose().eval() << "\n";
               });
}
}  // namespace uipc::backend::cuda