#include <affine_body/affine_body_diff_dof_reporter.h>
#include <affine_body/affine_body_diff_parm_reporter.h>
#include <utils/matrix_unpacker.h>
#include <utils/matrix_assembly_utils.h>
#include <affine_body/utils.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <affine_body/abd_diff_sim_manager.h>
#include <affine_body/constitutions/diff_mass_function.h>
#include <affine_body/affine_body_kinetic_diff_parm_reporter.h>
#include <affine_body/constitutions/diff_mass_function.h>
#include <affine_body/constitutions/abd_force_function.h>

namespace uipc::backend::cuda
{
// The Kinetic Energy Computation is already implemented in the abd_gradient_hessian_computer.cu file
// Maybe later we can move it here.

// AffineBodyKineticDiffDofReporter
//tex:
//deal with:
//
//$$
//\begin{bmatrix}
//\frac{\partial^2 E^{[k]}}{\partial X^{[k]} \partial X^{[k-2]}} &
//\frac{\partial^2 E^{[k]}}{\partial X^{[k]} \partial X^{[k-1]}}
//\end{bmatrix}
//$$
//
// Note: $$ \frac{\partial^2 E^{[k]}}{\partial X^{[k]} \partial X^{[k]}} $$
// is created directly by the global linear system, so we don't need to deal with it here.


//tex:
//$$
//G^{[k]} = M (q^{[k]} - 2q^{[k-1]} + q^{[k-2]} + G \cdot dt^2)
//$$
//
//$$ H^{[k][k - 1]} = -2M $$
//
//
//$$ H^{[k][k-2]} = M $$
class AffineBodyKineticDiffDofReporter : public AffineBodyDiffDofReporter
{

  public:
    using AffineBodyDiffDofReporter::AffineBodyDiffDofReporter;

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_report_extent(DiffDofExtentInfo& info) override
    {
        SizeT triplet_count = 0;
        auto  F             = info.frame();
        auto  body_count    = abd().body_count();

        SizeT M = (12 * 12) / (3 * 3);
        if(F == 2)
        {
            triplet_count = body_count * M;
        }
        else if(F >= 3)
        {
            triplet_count = body_count * M * 2;
        }

        info.triplet_count(triplet_count);
    }

    virtual void do_assemble(DiffDofInfo& info) override
    {
        using namespace muda;
        auto F          = info.frame();
        auto row_offset = info.dof_offset(F) / 3;
        auto row_count  = info.dof_count(F) / 3;

        auto triplet_offset = 0;

        if(F == 2)
        {
            // H^{[k][k-1]} = -2M

            auto Fm1 = F - 1;

            auto col_count  = info.dof_count(Fm1) / 3;
            auto col_offset = info.dof_offset(Fm1) / 3;

            auto N = abd().body_count();

            UIPC_ASSERT(row_count == col_count, "Now we don't support variant dof across frames");

            auto H = info.H();
            auto H_k_k_1 =
                H.subview(triplet_offset, N * 16).submatrix({row_offset, col_offset}, {row_count, col_count});

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(N,
                       [H_k_k_1 = H_k_k_1.viewer().name("H_k_k_1"),
                        masses  = abd().body_id_to_abd_mass.cviewer().name(
                            "masses")] __device__(int bodyI) mutable
                       {
                           Matrix12x12 H = -2 * masses(bodyI).to_mat();

                           // unpack 12x12 to 3x3
                           TripletMatrixUnpacker MU{H_k_k_1};

                           MU.block<4, 4>(bodyI * 16).write(bodyI * 4, bodyI * 4, H);
                       });

            triplet_offset += N * 16;
        }
        else if(F >= 3)
        {
            // H^{[k][k-1]} = -2M
            {
                auto Fm1 = F - 1;

                auto col_count  = info.dof_count(Fm1) / 3;
                auto col_offset = info.dof_offset(Fm1) / 3;
                auto N          = abd().body_count();

                UIPC_ASSERT(row_count == col_count,
                            "Now we don't support variant dof across frames");

                auto H = info.H();
                auto H_k_k_1 =
                    H.subview(triplet_offset, N * 16)
                        .submatrix({row_offset, col_offset}, {row_count, col_count});

                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(N,
                           [H_k_k_1 = H_k_k_1.viewer().name("H_k_k_1"),
                            masses  = abd().body_id_to_abd_mass.cviewer().name(
                                "masses")] __device__(int I) mutable
                           {
                               Matrix12x12 H = -2 * masses(I).to_mat();

                               TripletMatrixUnpacker MU{H_k_k_1};

                               MU.block<4, 4>(I * 16).write(I * 4, I * 4, H);
                           });

                triplet_offset += N * 16;
            }


            // H^{[k][k-2]} = M
            {
                auto Fm2 = F - 2;

                auto col_count  = info.dof_count(Fm2) / 3;
                auto col_offset = info.dof_offset(Fm2) / 3;

                auto N = abd().body_count();

                UIPC_ASSERT(row_count == col_count,
                            "Now we don't support variant dof across frames");

                auto H = info.H();
                auto H_k_k_2 =
                    H.subview(triplet_offset, N * 16)
                        .submatrix({row_offset, col_offset}, {row_count, col_count});

                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(N,
                           [H_k_k_2 = H_k_k_2.viewer().name("H_k_k_2"),
                            masses  = abd().body_id_to_abd_mass.cviewer().name(
                                "masses")] __device__(int I) mutable
                           {
                               Matrix12x12 H = masses(I).to_mat();

                               TripletMatrixUnpacker MU{H_k_k_2};

                               MU.block<4, 4>(I * 16).write(I * 4, I * 4, H);
                           });

                triplet_offset += N * 16;
            }
        }

        UIPC_ASSERT(triplet_offset == info.H().triplet_count(),
                    "triplet count mismatch, triplet_offset ({}) != H.triplet_count ({})",
                    triplet_offset,
                    info.H().triplet_count());
    }
};

REGISTER_SIM_SYSTEM(AffineBodyKineticDiffDofReporter);


// AffineBodyKineticDiffParmReporter
//tex:
//$$
// \begin{align}
// G =& M (q - \tilde{q}) \\
    // \tilde{q} =& q^n + \dot{q}_0 \cdot dt + ...\\
    // \frac{\partial G}{\partial \dot{q}_0} =& -M \cdot dt
// \end{align}
//$$
class AffineBodyKineticDiffInitVelocity final : public AffineBodyKineticDiffParmReporter
{
  public:
    using AffineBodyKineticDiffParmReporter::AffineBodyKineticDiffParmReporter;

    AffineBodyDynamics* affine_body_dynamics = nullptr;
    ABDDiffSimManager*  abd_diff_sim_manager = nullptr;

    vector<Vector12i> h_v0_parm_indices;
    vector<IndexT>    h_v0_indices;

    muda::DeviceBuffer<Vector12i> v0_parm_indices;
    muda::DeviceBuffer<IndexT>    v0_indices;

    virtual void do_build(BuildInfo& info) override
    {
        affine_body_dynamics = &require<AffineBodyDynamics>();
    }

    virtual void do_init() override
    {
        auto geo_slots = world().scene().geometries();

        using ForEachInfo = AffineBodyDynamics::ForEachInfo;

        affine_body_dynamics->for_each(
            geo_slots,
            [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                auto diff_velocity = sc.instances().find<Matrix4x4i>("diff/velocity");

                auto body_offset = I.geo_info().body_offset;

                auto diff_velocity_view = diff_velocity ? diff_velocity->view() :
                                                          span<const Matrix4x4i>{};
                for(auto&& [i, diff_parm_index] : enumerate(diff_velocity_view))
                {
                    if(diff_parm_index.maxCoeff() >= 0)  // if any parm index is valid
                    {
                        auto diff_index_of_q_v = transform_to_q(diff_parm_index);
                        h_v0_parm_indices.push_back(diff_index_of_q_v);
                        h_v0_indices.push_back(body_offset + i);
                    }
                }
            });

        v0_parm_indices.resize(h_v0_parm_indices.size());
        v0_parm_indices.view().copy_from(h_v0_parm_indices.data());

        v0_indices.resize(h_v0_indices.size());
        v0_indices.view().copy_from(h_v0_indices.data());
    }

    virtual void do_report_extent(DiffParmExtentInfo& info) override
    {
        // pGpqv0

        auto F = info.frame();

        if(F == 1)  // only at frame 1 we have v0
        {

            SizeT triplet_count = 0;
            // every q_v0 has 12 components, so we have 48 of matrix3x1
            triplet_count += h_v0_parm_indices.size() * 48;
            info.triplet_count(triplet_count);
        }
        else
        {
            info.triplet_count(0);
        }
    }

    virtual void do_update(GlobalDiffSimManager::DiffParmUpdateInfo& info) override
    {
        using namespace muda;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(h_v0_parm_indices.size(),
                   [parameters = info.parameters().viewer().name("parameters"),
                    v0_parm_indices = v0_parm_indices.cviewer().name("v0_parm_indices"),
                    v0_indices = v0_indices.cviewer().name("v0_indices"),
                    velocities = abd().body_id_to_q_v.viewer().name("velocities")] __device__(int I) mutable
                   {
                       const Vector12i& v0_parm_index = v0_parm_indices(I);
                       auto&            v0_index      = v0_indices(I);
                       Vector12         v0            = Vector12::Zero();
                       for(int i = 0; i < 12; i++)
                       {
                           auto parm_i = v0_parm_index(i);
                           if(parm_i >= 0)
                           {
                               v0(i) = parameters(parm_i);
                           }
                       }
                       velocities(v0_index) = v0;
                   });
    }

    virtual void do_assemble(DiffParmInfo& info) override
    {
        using namespace muda;

        auto F = info.frame();

        if(F == 1)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(h_v0_parm_indices.size(),
                       [v0_indices = v0_indices.cviewer().name("v0_indices"),
                        v0_parm_indices = v0_parm_indices.cviewer().name("v0_parm_indices"),
                        masses = abd().body_id_to_abd_mass.cviewer().name("masses"),
                        is_fixed = abd().body_id_to_is_fixed.cviewer().name("is_fixed"),
                        is_dynamic = abd().body_id_to_is_dynamic.cviewer().name("is_dynamic"),
                        pGpP = info.pGpP().viewer().name("pGpP"),
                        dt   = info.dt()] __device__(int I) mutable
                       {
                           const auto& m     = masses(I);
                           auto        index = v0_indices(I);

                           Vector12i   parm_index = v0_parm_indices(I);
                           Matrix12x12 pGpQv;

                           if(is_fixed(index))
                           {
                               pGpQv.setZero();
                           }
                           else
                           {
                               pGpQv = (-dt) * m.to_mat();
                           }

                           IndexT max_parm_index = parm_index.maxCoeff();

                           MUDA_ASSERT(max_parm_index >= 0,
                                       "max_parm_index (%d) < 0, why can it happen?",
                                       max_parm_index);

                           // unpack 12x12 to 3x1
                           TripletMatrixUnpacker MU{pGpP};

                           // fill one col every time
                           for(int i = 0; i < 12; i++)
                           {
                               // if parm_i < 0, then igonre this col;

                               auto parm_i = parm_index(i);

                               if(parm_i >= 0)
                               {
                                   MU.block<4, 1>(I * 48 + i * 4)
                                       .write(I * 4, parm_i, pGpQv.col(i));
                               }
                               else  // parm to ignore
                               {
                                   // write to a safe place (max_parm_index) with Zero Matrix
                                   // equivalent to no effect

                                   MU.block<4, 1>(I * 48 + i * 4)
                                       .write(I * 12, max_parm_index, Vector12::Zero());
                               }
                           }
                       });
        }
        else
        {
            UIPC_ASSERT(info.pGpP().triplet_count() == 0, "frame!=1, no diff parm needed.");
        }
    }
};

REGISTER_SIM_SYSTEM(AffineBodyKineticDiffInitVelocity);


// AffineBodyKineticDiffRestPosition
class AffineBodyKineticDiffRestPosition final : public AffineBodyKineticDiffParmReporter
{
  public:
    using AffineBodyKineticDiffParmReporter::AffineBodyKineticDiffParmReporter;

    AffineBodyDynamics* affine_body_dynamics = nullptr;
    ABDDiffSimManager*  abd_diff_sim_manager = nullptr;

    vector<Vector12i> h_mass_parm_indices;
    vector<IndexT>    h_mass_indices;

    muda::DeviceBuffer<Vector12i> mass_parm_indices;
    muda::DeviceBuffer<IndexT>    mass_indices;

    virtual void do_build(BuildInfo& info) override
    {
        affine_body_dynamics = &require<AffineBodyDynamics>();
        abd_diff_sim_manager = &require<ABDDiffSimManager>();
    }

    virtual void do_init() override {}

    virtual void do_update(DiffParmUpdateInfo& info) override {}

    virtual void do_report_extent(DiffParmExtentInfo& info) override
    {
        auto tri_size = info.diff_triangles().size();
        auto tet_size = info.diff_tetrahedra().size();
        UIPC_ASSERT(tet_size == 0, "Now we only support diff triangle mesh");

        // a triangle has 3 vertices, each vertex has 3 components, so we have 9 parameters (columns)
        // M * (q - \tilde{q}) has 12 components (rows)
        // so the local jacobian is a 12x9 matrix => (4 * matrix3x1) * 9

        SizeT triplet_count = tri_size * 4 * 9;
        info.triplet_count(triplet_count);
    }

    virtual void do_assemble(DiffParmInfo& info) override
    {
        using namespace muda;
        auto diff_triangles = abd_diff_sim_manager->diff_triangles();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                diff_triangles.size(),
                [diff_triangles = diff_triangles.cviewer().name("diff_triangles"),
                 diff_rest_position =
                     abd_diff_sim_manager->diff_rest_positions().viewer().name("diff_rest_position"),
                 Js       = info.Js().viewer().name("Js"),
                 pGpPs    = info.pGpP().viewer().name("pGpPs"),
                 Ms       = info.body_masses().viewer().name("Ms"),
                 qs       = info.qs().viewer().name("qs"),
                 q_vs     = info.q_vs().viewer().name("q_vs"),
                 q_tildes = info.q_tildes().viewer().name("q_tildes"),
                 q_prevs  = info.q_prevs().viewer().name("q_prevs"),
                 v2b      = info.v2b().viewer().name("v2b"),
                 rhos = abd_diff_sim_manager->body_mass_density().viewer().name("rhos"),
                 gravities = abd_diff_sim_manager->instance_gravity().viewer().name("gravities"),
                 dt = info.dt()] __device__(int I) mutable
                {
                    Vector3i T = diff_triangles(I);

                    Vector9 X0;
                    X0.segment<3>(0) = Js(T(0)).x_bar();
                    X0.segment<3>(3) = Js(T(1)).x_bar();
                    X0.segment<3>(6) = Js(T(2)).x_bar();

                    Vector9i diff_X0;
                    diff_X0.segment<3>(0) = diff_rest_position(T(0));
                    diff_X0.segment<3>(3) = diff_rest_position(T(1));
                    diff_X0.segment<3>(6) = diff_rest_position(T(2));

                    MUDA_ASSERT(v2b(T(0)) == v2b(T(1)) && v2b(T(0)) == v2b(T(2)),
                                "Now we only support same body triangle, but got triangle (%d,%d,%d), body (%d,%d,%d)",
                                T(0),
                                T(1),
                                T(2),
                                v2b(T(0)),
                                v2b(T(1)),
                                v2b(T(2)));

                    //tex:
                    //$$
                    //\begin{align}
                    //G =& M (q - \tilde{q}) \\
                    //\frac{\partial G}{\partial \bar{x_i}} =& \frac{\partial M (q - \tilde{q})}{\partial \bar{x_i}} \\
                    // =& \frac{\partial M}{\partial \bar{x_i}} (q - q^t - \dot{q}^t \cdot \Delta t) - \frac{\partial f}{\partial \bar{x_i}} \Delta t^2
                    //\end{align}
                    //$$
                    IndexT   bI = v2b(T(0));
                    Vector12 dq = qs(bI) - q_prevs(bI) - q_vs(bI) * dt;

                    Float   rho = rhos(bI);
                    Vector3 Ig  = gravities(bI);
                    // TODO: Now we only assume the external force is gravity
                    // later we need to consider other forces and provide new class
                    // to handle the diff force part (for generality)
                    Vector3 force_density = rho * Ig;
                    // printf("fd: %f %f %f\n", force_density(0), force_density(1), force_density(2));

                    Eigen::Matrix<Float, 12, 9> pGpP;
                    sym::diff_m::pGpX0(pGpP, X0, dq, rho);  // left part
                    Eigen::Matrix<Float, 12, 9> df;
                    sym::force::diffF(df, X0, force_density);  // right part
                    Float dt2 = dt * dt;
                    pGpP -= df * dt2;

                    auto default_index = safe_diff_parm(diff_X0);

                    auto TMU = TripletMatrixUnpacker{pGpPs};

                    auto triplet_offset = I * 4 * 9;  // 36 * matrix3x1
                    auto row            = bI * 4;     // 4 = 12 / 3

                    // Every time we write a colunm of pGpP
                    for(int i = 0; i < diff_X0.size(); ++i)
                    {
                        IndexT   col;
                        Vector12 pGpP_col;
                        if(diff_X0(i) >= 0)
                        {
                            col      = diff_X0(i);
                            pGpP_col = pGpP.col(i);

                            //cout << "Kinetic T:" << T.transpose().eval()
                            //     << "pGpX0[" << diff_X0(i) << "]:\n"
                            //     << pGpP_col.transpose().eval() << "\n";
                        }
                        else
                        {
                            col = default_index;
                            pGpP_col.setZero();
                        }

                        // take a range [triplet_offset, triplet_offset + 4) of matrix3x1
                        TMU.block<4, 1>(triplet_offset)  //
                            .write(row,                  // start row
                                   col,                  // start col
                                   pGpP_col  // write a column of pGpP
                            );

                        triplet_offset += 4;  // go next range
                    }
                });
    }
};

REGISTER_SIM_SYSTEM(AffineBodyKineticDiffRestPosition);
}  // namespace uipc::backend::cuda