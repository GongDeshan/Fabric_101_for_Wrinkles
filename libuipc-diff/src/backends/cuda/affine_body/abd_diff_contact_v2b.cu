#include <affine_body/affine_body_diff_parm_reporter.h>
#include <affine_body/abd_contact_receiver.h>
#include <contact_system/global_contact_manager.h>
#include <utils/matrix_assembly_utils.h>
#include <utils/matrix_unpacker.h>
#include <muda/cub/device/device_scan.h>
#include <kernel_cout.h>

namespace uipc::backend::cuda::sym::diff_v2b
{
#include <affine_body/constitutions/sym/diffG12.inl>
}


namespace uipc::backend::cuda
{
class AbdDiffContactV2B final : public AffineBodyDiffParmReporter
{
  public:
    using AffineBodyDiffParmReporter::AffineBodyDiffParmReporter;
    GlobalContactManager*      contact_manager = nullptr;
    AffineBodyVertexReporter*  vertex_reporter = nullptr;
    ABDContactReceiver*        receiver        = nullptr;
    muda::DeviceBuffer<IndexT> is_diff_vert;
    muda::DeviceBuffer<IndexT> is_diff_vert_offsets;

    muda::DeviceBuffer<IndexT> diff_vert_contact_gradient;
    muda::DeviceBuffer<IndexT> diff_vert_contact_hessian;

    virtual void do_build(BuildInfo& info) override
    {
        contact_manager = &require<GlobalContactManager>();
        receiver        = &require<ABDContactReceiver>();
        vertex_reporter = &require<AffineBodyVertexReporter>();
    }

    virtual void do_init() override {}

    virtual void do_update(GlobalDiffSimManager::DiffParmUpdateInfo& info) override
    {
    }

    IndexT select_diff_vert_grad(DiffParmExtentInfo& info)
    {
        if(info.diff_vertices().size() == 0)
        {
            diff_vert_contact_gradient.resize(0);
            return 0;
        }

        auto vert_offset = vertex_reporter->vertex_offset();
        auto grad        = receiver->contact_gradient();
        auto grad_count  = receiver->contact_gradient().doublet_count();

        is_diff_vert.resize(grad_count + 1);
        is_diff_vert_offsets.resize(grad_count + 1);  // +1 for the total count

        using namespace muda;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(grad_count,
                   [diff_rest_position = info.diff_rest_positions().viewer().name("diff_rest_position"),
                    grad         = grad.viewer().name("grad"),
                    is_diff_vert = is_diff_vert.viewer().name("is_diff_vert"),
                    is_diff_vert_offsets = is_diff_vert_offsets.viewer().name("is_diff_vert_offsets"),
                    vert_offset,
                    grad_count] __device__(int I) mutable
                   {
                       auto&& [global_vI, g]    = grad(I);
                       IndexT          vI       = global_vI - vert_offset;
                       const Vector3i& diff_pos = diff_rest_position(vI);

                       if(diff_pos.maxCoeff() >= 0)
                       {
                           is_diff_vert(I) = 1;
                       }
                       else
                       {
                           is_diff_vert(I) = 0;
                       }

                       if(I == 0)
                       {
                           is_diff_vert_offsets(grad_count) = 0;  // set the last element to 0
                       }
                   });

        muda::DeviceScan().ExclusiveSum(
            is_diff_vert.data(), is_diff_vert_offsets.data(), grad_count + 1);

        IndexT total_count;
        is_diff_vert_offsets.view(grad_count).copy_to(&total_count);

        diff_vert_contact_gradient.resize(total_count);

        spdlog::info("Affine Body Diff Vertices In Contact Gradient: {}", total_count);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(grad_count,
                   [is_diff_vert = is_diff_vert.viewer().name("is_diff_vert"),
                    is_diff_vert_offsets = is_diff_vert_offsets.viewer().name("is_diff_vert_offsets"),
                    grad = grad.viewer().name("grad"),
                    diff_vert_contact_gradient =
                        diff_vert_contact_gradient.viewer().name("diff_vert_contact_gradient"),
                    vert_offset] __device__(int I) mutable
                   {
                       auto&& [global_vI, g] = grad(I);
                       IndexT vI             = global_vI - vert_offset;

                       if(is_diff_vert(I))
                       {
                           auto offset = is_diff_vert_offsets(I);
                           diff_vert_contact_gradient(offset) = I;
                       }
                   });

        return total_count;
    }
    IndexT select_diff_vert_hessian(DiffParmExtentInfo& info)
    {

        if(info.diff_vertices().size() == 0)
        {
            diff_vert_contact_hessian.resize(0);
            return 0;
        }

        auto vert_offset   = vertex_reporter->vertex_offset();
        auto hessian       = receiver->contact_hessian();
        auto hessian_count = receiver->contact_hessian().triplet_count();

        is_diff_vert.resize(hessian_count + 1);
        is_diff_vert_offsets.resize(hessian_count + 1);  // +1 for the total count

        using namespace muda;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(hessian_count,
                   [diff_rest_position = info.diff_rest_positions().viewer().name("diff_rest_position"),
                    hessian      = hessian.viewer().name("hessian"),
                    is_diff_vert = is_diff_vert.viewer().name("is_diff_vert"),
                    is_diff_vert_offsets = is_diff_vert_offsets.viewer().name("is_diff_vert_offsets"),
                    vert_offset,
                    hessian_count] __device__(int I) mutable
                   {
                       auto&& [global_i, global_j, H] = hessian(I);
                       IndexT          vI             = global_i - vert_offset;
                       bool            is_diag        = global_i == global_j;
                       const Vector3i& diff_pos       = diff_rest_position(vI);

                       if(diff_pos.maxCoeff() >= 0 && is_diag)  // only the diagonal elements
                       {
                           is_diff_vert(I) = 1;
                       }
                       else
                       {
                           is_diff_vert(I) = 0;
                       }

                       if(I == 0)
                       {
                           is_diff_vert_offsets(hessian_count) = 0;  // set the last element to 0
                       }
                   });

        muda::DeviceScan().ExclusiveSum(
            is_diff_vert.data(), is_diff_vert_offsets.data(), hessian_count + 1);

        IndexT total_count;
        is_diff_vert_offsets.view(hessian_count).copy_to(&total_count);

        diff_vert_contact_hessian.resize(total_count);

        spdlog::info("Affine Body Diff Vertices In Contact Hessian: {}", total_count);

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(hessian_count,
                   [is_diff_vert = is_diff_vert.viewer().name("is_diff_vert"),
                    is_diff_vert_offsets = is_diff_vert_offsets.viewer().name("is_diff_vert_offsets"),
                    hessian = hessian.viewer().name("hessian"),
                    diff_vert_contact_hessian =
                        diff_vert_contact_hessian.viewer().name("diff_vert_contact_hessian"),
                    vert_offset] __device__(int I) mutable
                   {
                       auto&& [global_i, global_j, H] = hessian(I);
                       IndexT vI                      = global_i - vert_offset;

                       if(is_diff_vert(I))
                       {
                           auto offset = is_diff_vert_offsets(I);
                           diff_vert_contact_hessian(offset) = I;
                       }
                   });

        return total_count;
    }


    virtual void do_report_extent(DiffParmExtentInfo& info) override
    {
        auto count  = select_diff_vert_grad(info);
        auto count_ = select_diff_vert_hessian(info);
        UIPC_ASSERT(count == count_, "grad count({}) != diag hessian count_({})", count, count_);

        // A vertex has 3 components, so we have 3 parameters
        // J.T*G(x) has 12 components
        // so we have 12 rows
        // so the jacobian is a 12x3 matrix => (4 * matrix3x1) * 3
        SizeT triplet_count = count * 4 * 3;
        info.triplet_count(triplet_count);
    }

    virtual void do_assemble(DiffParmInfo& info) override
    {
        using namespace muda;
        auto count = diff_vert_contact_gradient.size();
        if(count == 0)
            return;

        auto vert_offset      = vertex_reporter->vertex_offset();
        auto contact_gradient = receiver->contact_gradient();
        auto contact_hessian  = receiver->contact_hessian();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(count,
                   [pGpPs = info.pGpP().viewer().name("pGpPs"),
                    diff_vert_contact_gradient =
                        diff_vert_contact_gradient.viewer().name("diff_vert_contact_gradient"),
                    diff_vert_contact_hessian =
                        diff_vert_contact_hessian.viewer().name("diff_vert_contact_hessian"),
                    diff_vert = info.diff_vertices().viewer().name("diff_vertices"),
                    volumes = info.body_volumes().viewer().name("volumes"),
                    diff_rest_position = info.diff_rest_positions().viewer().name("diff_rest_position"),
                    contact_gradient = contact_gradient.viewer().name("contact_gradient"),
                    contact_hessian = contact_hessian.viewer().name("contact_hessian"),
                    v2b         = info.v2b().viewer().name("v2b"),
                    Js          = info.Js().viewer().name("Js"),
                    qs          = info.qs().viewer().name("qs"),
                    vert_offset = vert_offset] __device__(int I) mutable
                   {
                       IndexT diff_vert_grad = diff_vert_contact_gradient(I);
                       IndexT diff_vert_hess = diff_vert_contact_hessian(I);

                       // Because the contact gradient and hessian are sorted by the vertex index
                       // so the global_vI should be the same
                       // e.g.
                       // G[1] -> H[1,1]
                       // G[2] -> H[2,2]
                       // G[3] -> H[3,3]
                       // ...
                       auto&& [global_vI, g] = contact_gradient(diff_vert_grad);
                       auto&& [global_i, global_j, H] = contact_hessian(diff_vert_hess);

                       MUDA_ASSERT(global_vI == global_i,
                                   "global_vI(%d) != global_i(%d)",
                                   global_vI,
                                   global_i);

                       IndexT   vI            = global_vI - vert_offset;
                       IndexT   bI            = v2b(vI);
                       Vector3i diff_X0       = diff_rest_position(vI);
                       IndexT   default_index = safe_diff_parm(diff_X0);

                       Matrix<Float, 12, 3> pGpX0;
                       sym::diff_v2b::pG12px0(pGpX0, Js(vI).x_bar(), g, H, qs(bI));

                       auto row            = bI * 4;
                       auto triplet_offset = I * 3 * 3;

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

                               //cout << " V2B pGpX0[" << diff_X0(i) << "]:\n"
                               //     << pGpX0_col.transpose().eval() << "\n";
                           }
                           else
                           {
                               col       = default_index;
                               pGpX0_col = Vector12::Zero();
                           }

                           // A col
                           // 0_3x1 (ignored)
                           // []_{3x1}
                           // []_{3x1}
                           // []_{3x1}
                           TMU.block<4, 1>(triplet_offset)  // a range from [triplet_offset, triplet_offset + 3) of matrix3x1
                               .write(row,  // row start from bI * 4 + 1
                                      col,  // col is the parm index
                                      pGpX0_col);

                           triplet_offset += 4;
                       }
                   });
    }
};

REGISTER_SIM_SYSTEM(AbdDiffContactV2B);
}  // namespace uipc::backend::cuda
