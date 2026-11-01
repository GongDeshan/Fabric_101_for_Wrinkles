#include <affine_body/abd_contact_diff_parm_reporter.h>
#include <utils/matrix_unpacker.h>
#include <kernel_cout.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(ABDContactDiffParmReporter);


void ABDContactDiffParmReporter::do_build(BuildInfo& info)
{
    m_impl.abd             = &require<AffineBodyDynamics>();
    m_impl.vertex_reporter = &require<AffineBodyVertexReporter>();
    m_impl.receiver        = &require<ABDContactDiffParmReceiver>();
}

void ABDContactDiffParmReporter::do_report_extent(DiffParmExtentInfo& info)
{
    // contact hessian matrix is H3x3 of vertex-pair (i,j)
    auto triplet_count = m_impl.receiver->pGpP().triplet_count();

    // affine body pGpP is 12x1 matrix J^T * pGpX => 4 * matrix3x1
    info.triplet_count(triplet_count * 4);
}

void ABDContactDiffParmReporter::do_assemble(DiffParmInfo& info)
{
    m_impl.assemble(info);
}

void ABDContactDiffParmReporter::do_update(GlobalDiffSimManager::DiffParmUpdateInfo& info)
{
}

void ABDContactDiffParmReporter::do_init() {}

void ABDContactDiffParmReporter::Impl::assemble(DiffParmInfo& info)
{
    using namespace muda;

    auto global_vertex_offset = vertex_reporter->vertex_offset();

    auto contact_pGpP = receiver->pGpP();
    auto pGpP         = info.pGpP();

    if(pGpP.triplet_count() == 0)  // ignore empty pGpP
        return;

    UIPC_ASSERT(contact_pGpP.total_triplet_count() > 0,
                "contact_pGpP is empty, but pGpP Count = {}",
                pGpP.triplet_count());

    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(contact_pGpP.triplet_count(),
               [contact_pGpPs = contact_pGpP.viewer().name("contact_pGpPs"),
                v2b           = abd->v2b().cviewer().name("v2b"),
                Js            = abd->Js().viewer().name("Js"),
                body_is_fixed = abd->body_is_fixed().viewer().name("body_is_fixed"),
                global_vertex_offset,
                pGpPs = pGpP.viewer().name("pGpPs")] __device__(int I) mutable
               {
                   auto&& [global_i, parm_j, contact_pGpP] = contact_pGpPs(I);
                   TripletMatrixUnpacker TMU{pGpPs};

                   auto i     = global_i - global_vertex_offset;
                   auto parmJ = parm_j;

                   const ABDJacobi& Ji = Js(i);

                   Vector12 pGpP;

                   auto bodyI = v2b(i);


                   if(body_is_fixed(bodyI))
                   {
                       pGpP.setZero();
                   }
                   else
                   {
                       pGpP = Ji.T() * contact_pGpP;
                   }

                   TMU.block<4, 1>(4 * I).write(  //
                       bodyI * 4,                 // i
                       parmJ,                     // parmJ
                       pGpP                       // pGpP
                   );

                   // cout << "Body pGpP: " << pGpP.transpose().eval() << "\n";
               });
}
}  // namespace uipc::backend::cuda