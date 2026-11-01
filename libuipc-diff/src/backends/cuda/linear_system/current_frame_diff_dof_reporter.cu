#include <diff_sim/diff_dof_reporter.h>
#include <linear_system/global_linear_system.h>
#include <utils/matrix_unpacker.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>

namespace uipc::backend::cuda
{
/**
 * @brief Compute the Diff Dof for the current frame
 * 
 * This class directly take the result from the global linear system,
 * because the global linear system's full hessian matrix is exactly the 
 * 
 * $$
 * \frac{\partial^2 E}{\partial X^{[i]} \partial X^{[i]}}
 * $$
 * 
 * where $^{[i]}$ is the i-th frame
 */
class CurrentFrameDiffDofReporter : public DiffDofReporter
{
  public:
    using DiffDofReporter::DiffDofReporter;

    GlobalLinearSystem* global_linear_system = nullptr;
    IndexT              triplet_count        = 0;

    muda::BCOOMatrixView<Float, 3> bcoo_A()
    {
        return global_linear_system->m_impl.bcoo_A;
    }

    virtual void do_build(BuildInfo& info) override
    {
        global_linear_system = &require<GlobalLinearSystem>();
    }

    virtual void do_report_extent(GlobalDiffSimManager::DiffDofExtentInfo& info) override
    {
        auto A        = bcoo_A();
        triplet_count = A.triplet_count();

        // report the triplet count
        info.triplet_count(triplet_count);
    }

    virtual void do_assemble(GlobalDiffSimManager::DiffDofInfo& info) override
    {
        using namespace muda;


        auto F          = info.frame();
        auto dof_offset = info.dof_offset(F) / 3;
        auto dof_count  = info.dof_count(F) / 3;

        // take the diagonal submatrix of total H
        auto sub_H = info.H().submatrix({dof_offset, dof_offset}, {dof_count, dof_count});
        auto A = bcoo_A();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(A.triplet_count(),
                   [bcoo_A = A.cviewer().name("bcoo_A"),
                    sub_H = sub_H.viewer().name("sub_H")] __device__(int I) mutable
                   {
                       auto&& [i, j, H3x3] = bcoo_A(I);
                       sub_H(I).write(i, j, H3x3);
                   });
    }
};

// NOW: we don't need to fill the current frame's diff dof anymore
// just keep the code here for reference
// removing the registration

//REGISTER_SIM_SYSTEM(CurrentFrameDiffDofReporter);
}  // namespace uipc::backend::cuda
