#include <affine_body/affine_body_constitution_diff_volume.h>
#include <affine_body/abd_gradient_hessian_computer.h>
#include <affine_body/abd_diff_sim_manager.h>
#include <utils/matrix_assembly_utils.h>
#include <utils/matrix_unpacker.h>
#include <affine_body/constitutions/diff_volume_function.h>
#include <kernel_cout.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(AffineBodyConstitutionDiffVolume);

//tex:
// Because All AffineBodyConstitutions have: $E = V \cdot I(A)$
// So the pGpV will be:
// $$
// \frac{\partial G}{\partial V} = \frac{\partial V\cdot I(A)}{\partial V} = \frac{1}{V} \cdot G
// $$

void AffineBodyConstitutionDiffVolume::do_build(BuildInfo& info)
{
    m_impl.abd_gradient_hessian_computer = &require<ABDGradientHessianComputer>();
    m_impl.abd_diff_sim_manager = &require<ABDDiffSimManager>();
}

void AffineBodyConstitutionDiffVolume::do_init() {}

void AffineBodyConstitutionDiffVolume::do_update(DiffParmUpdateInfo& info) {}

void AffineBodyConstitutionDiffVolume::do_report_extent(DiffParmExtentInfo& info)
{
    auto tri_size = info.diff_triangles().size();
    auto tet_size = info.diff_tetrahedra().size();
    UIPC_ASSERT(tet_size == 0, "Now we only support diff triangle mesh");

    // a triangle has 3 vertices, each vertex has 3 components, so we have 9 parameters
    // pGpV has 12 components, so we have 12 rows
    // so the local jacobian is a 12x9 matrix => (4 * matrix3x1) * 9

    SizeT triplet_count = tri_size * 4 * 9;
    info.triplet_count(triplet_count);
}

void AffineBodyConstitutionDiffVolume::Impl::assemble(DiffParmInfo& info)
{
    using namespace muda;

    auto tri_count = info.diff_triangles().size();
    auto shape_graidient = abd_gradient_hessian_computer->m_impl.shape_gradient.view();
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(tri_count,
               [pGpPs = info.pGpP().viewer().name("pGpPs"),
                diff_triangles = info.diff_triangles().viewer().name("diff_triangles"),
                shape_gradient = shape_graidient.viewer().name("shape_gradient"),
                volumes = info.body_volumes().viewer().name("volumes"),
                diff_rest_position = info.diff_rest_positions().viewer().name("diff_rest_position"),
                v2b = info.v2b().viewer().name("v2b"),
                Js  = info.Js().viewer().name("Js")] __device__(int I) mutable
               {
                   Vector3i T = diff_triangles(I);
                   Vector9  X0;
                   X0.segment<3>(0) = Js(T(0)).x_bar();
                   X0.segment<3>(3) = Js(T(1)).x_bar();
                   X0.segment<3>(6) = Js(T(2)).x_bar();

                   Vector9i diff_X0;

                   diff_X0.segment<3>(0) = diff_rest_position(T(0));
                   diff_X0.segment<3>(3) = diff_rest_position(T(1));
                   diff_X0.segment<3>(6) = diff_rest_position(T(2));

                   IndexT default_index = safe_diff_parm(diff_X0);

                   IndexT bI = v2b(T(0));
                   Float  V  = volumes(bI);

                   Vector12 G = shape_gradient(bI) / V;

                   Vector9 pVpX0;
                   sym::diff_v::pVpX0(pVpX0, X0);

                   Matrix<Float, 12, 9> pGpX0 = G * pVpX0.transpose();

                   auto row            = bI * 4;
                   auto triplet_offset = I * 4 * 9;

                   auto TMU = TripletMatrixUnpacker{pGpPs};

                   // Every time we write a col of pGpX0
                   for(int i = 0; i < diff_X0.size(); ++i)
                   {
                       IndexT   col;
                       Vector12 pGpX0_col;
                       if(diff_X0(i) >= 0)
                       {
                           col       = diff_X0(i);
                           pGpX0_col = pGpX0.col(i);
                       }
                       else
                       {
                           col       = default_index;
                           pGpX0_col = Vector12::Zero();
                       }

                       // A col
                       // []_{3x1}
                       // []_{3x1}
                       // []_{3x1}
                       // []_{3x1}
                       TMU.block<4, 1>(triplet_offset)  // a range from [triplet_offset, triplet_offset + 4) of matrix3x1
                           .write(row,  // row start from bI * 4
                                  col,  // col is the parm index
                                  pGpX0_col);


                       triplet_offset += 4;
                   }
               })
        .wait();
}

void AffineBodyConstitutionDiffVolume::do_assemble(DiffParmInfo& info)
{
    m_impl.assemble(info);
}
}  // namespace uipc::backend::cuda
