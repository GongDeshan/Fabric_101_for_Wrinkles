#include <finite_element/finite_element_kinetic.h>
#include <finite_element/finite_element_diff_dof_reporter.h>
#include <finite_element/finite_element_diff_parm_reporter.h>
#include <utils/matrix_unpacker.h>

namespace uipc::backend::cuda
{
void FiniteElementKinetic::do_build(BuildInfo& info)
{
    m_impl.finite_element_method = &require<FiniteElementMethod>();
    m_impl.finite_element_method->add_constitution(this);
}

void FiniteElementKinetic::do_report_extent(ReportExtentInfo& info)
{
    auto vert_count = m_impl.fem().xs.size();
    info.energy_count(vert_count);
    info.stencil_dim(1);
}

void FiniteElementKinetic::do_compute_energy(ComputeEnergyInfo& info)
{
    m_impl.compute_energy(info);
}

void FiniteElementKinetic::do_compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    m_impl.compute_gradient_hessian(info);
}

void FiniteElementKinetic::Impl::compute_energy(ComputeEnergyInfo& info)
{
    using namespace muda;
    // Compute kinetic energy
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(fem().xs.size(),
               [is_fixed = fem().is_fixed.cviewer().name("is_fixed"),
                xs       = fem().xs.cviewer().name("xs"),
                x_tildes = fem().x_tildes.viewer().name("x_tildes"),
                masses   = fem().masses.cviewer().name("masses"),
                Ks = info.energies().viewer().name("kinetic_energy")] __device__(int i) mutable
               {
                   auto& K = Ks(i);
                   if(is_fixed(i))
                   {
                       K = 0.0;
                   }
                   else
                   {
                       const Vector3& x       = xs(i);
                       const Vector3& x_tilde = x_tildes(i);
                       Float          M       = masses(i);
                       Vector3        dx      = x - x_tilde;
                       K                      = 0.5 * M * dx.dot(dx);
                   }
               });
}
void FiniteElementKinetic::Impl::compute_gradient_hessian(ComputeGradientHessianInfo& info)
{
    using namespace muda;

    // Kinetic
    ParallelFor()
        .file_line(__FILE__, __LINE__)
        .apply(fem().xs.size(),
               [xs         = fem().xs.viewer().name("xs"),
                x_tildes   = fem().x_tildes.cviewer().name("x_tildes"),
                vs         = fem().vs.viewer().name("vs"),
                masses     = fem().masses.cviewer().name("masses"),
                is_fixed   = fem().is_fixed.cviewer().name("is_fixed"),
                is_dynamic = fem().is_dynamic.cviewer().name("is_dynamic"),
                G3s        = info.gradients().viewer().name("G3s"),
                H3x3s = info.hessians().viewer().name("H3x3s")] __device__(int i) mutable
               {
                   auto& m       = masses(i);
                   auto& x       = xs(i);
                   auto& x_tilde = x_tildes(i);

                   Vector3   G;
                   Matrix3x3 H;

                   if(is_fixed(i))  // fixed
                   {
                       G = Vector3::Zero();
                   }
                   else
                   {
                       G = m * (x - x_tilde);
                   }

                   H = masses(i) * Matrix3x3::Identity();

                   G3s(i).write(i, G);
                   H3x3s(i).write(i, i, H);
               });
}

REGISTER_SIM_SYSTEM(FiniteElementKinetic);

class FiniteElementKineticDiffDofReporter : public FiniteElementDiffDofReporter
{
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


  public:
    using FiniteElementDiffDofReporter::FiniteElementDiffDofReporter;

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_report_extent(DiffDofExtentInfo& info) override
    {
        SizeT triplet_count = 0;
        auto  F             = info.frame();

        if(F == 2)
        {
            auto Fm1      = F - 1;
            triplet_count = info.dof_count(Fm1) / 3;
        }
        else if(F >= 3)
        {
            auto Fm1      = F - 1;
            auto Fm2      = F - 2;
            triplet_count = info.dof_count(Fm1) / 3 + info.dof_count(Fm2) / 3;
        }

        info.triplet_count(triplet_count);
    }

    virtual void do_assemble(DiffDofInfo& info) override
    {
        using namespace muda;
        auto F          = info.frame();
        auto row_offset = info.dof_offset(F) / 3;
        auto row_count  = info.dof_count(F) / 3;
        auto N          = row_count;

        auto triplet_offset = 0;

        if(F == 2)
        {
            // H^{[k][k-1]} = -2M

            auto Fm1 = F - 1;

            auto col_count  = info.dof_count(Fm1) / 3;
            auto col_offset = info.dof_offset(Fm1) / 3;

            UIPC_ASSERT(row_count == col_count, "Now we don't support variant dof across frames");

            auto H = info.H();
            auto H_k_k_1 =
                H.subview(triplet_offset, col_count)
                    .submatrix({row_offset, col_offset}, {row_count, col_count});

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(N,
                       [H_k_k_1 = H_k_k_1.viewer().name("H_k_k_1"),
                        masses = fem().masses.cviewer().name("masses")] __device__(int I) mutable
                       {
                           //tex:
                           //$$
                           //G^{[2]} = m (x^{[2]} - 2x^{[1]} + x^{[0]} + g \cdot dt^2)
                           //$$
                           //$$ H^{[2][1]} = -2m $$

                           auto m = masses(I);
                           H_k_k_1(I).write(I, I, -2.0 * m * Matrix3x3::Identity());
                       });

            triplet_offset = col_count;
        }
        else if(F >= 3)
        {
            // H^{[k][k-1]} = -2M
            {
                auto Fm1 = F - 1;

                auto col_count  = info.dof_count(Fm1) / 3;
                auto col_offset = info.dof_offset(Fm1) / 3;

                UIPC_ASSERT(row_count == col_count,
                            "Now we don't support variant dof across frames");

                auto H = info.H();
                auto H_k_k_1 =
                    H.subview(triplet_offset, col_count)
                        .submatrix({row_offset, col_offset}, {row_count, col_count});

                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(N,
                           [H_k_k_1 = H_k_k_1.viewer().name("H_k_k_1"),
                            masses = fem().masses.cviewer().name("masses")] __device__(int I) mutable
                           {
                               //tex:
                               //$$
                               //G^{[k]} = m (x^{[k]} - 2x^{[k-1]} + x^{[k-2]} + g \cdot dt^2)
                               //$$
                               //$$ H^{[k][k-1]} = -2m $$

                               auto m = masses(I);
                               H_k_k_1(I).write(I, I, -2.0 * m * Matrix3x3::Identity());
                           });

                triplet_offset += col_count;
            }


            // H^{[k][k-2]} = M
            {
                auto Fm2 = F - 2;

                auto col_count  = info.dof_count(Fm2) / 3;
                auto col_offset = info.dof_offset(Fm2) / 3;

                // spdlog::info("F: {}, col_count: {}, col_offset: {}", F, col_count, col_offset);

                UIPC_ASSERT(row_count == col_count,
                            "Now we don't support variant dof across frames");

                auto H = info.H();
                auto H_k_k_2 =
                    H.subview(triplet_offset, col_count)
                        .submatrix({row_offset, col_offset}, {row_count, col_count});

                ParallelFor()
                    .file_line(__FILE__, __LINE__)
                    .apply(N,
                           [H_k_k_2 = H_k_k_2.viewer().name("H_k_k_2"),
                            masses = fem().masses.cviewer().name("masses")] __device__(int I) mutable
                           {
                               //tex:
                               //$$
                               //G^{[k]} = m (x^{[k]} - 2x^{[k-1]} + x^{[k-2]} + g \cdot dt^2)
                               //$$
                               //$$ H^{[k][k-2]} = m $$

                               auto m = masses(I);
                               H_k_k_2(I).write(I, I, m * Matrix3x3::Identity());
                           });

                triplet_offset += col_count;
            }
        }

        UIPC_ASSERT(triplet_offset == info.H().triplet_count(), "triplet count mismatch");
    }
};

REGISTER_SIM_SYSTEM(FiniteElementKineticDiffDofReporter);

class FiniteElementKineticDiffParmReporter final : public FiniteElementDiffParmReporter
{
    //tex:
    //$$
    // \begin{align}
    // G =& m (x - \tilde{x}) \\
    // \tilde{x} =& x^n + v_0 \cdot dt + ...\\
    // \frac{\partial G}{\partial v_0} =& -m \cdot dt
    //\end{align}
    //$$
  public:
    using FiniteElementDiffParmReporter::FiniteElementDiffParmReporter;

    FiniteElementMethod* fem = nullptr;

    vector<Vector3i> h_v0_parm_indices;
    vector<IndexT>   h_v0_indices;

    muda::DeviceBuffer<Vector3i> v0_parm_indices;
    muda::DeviceBuffer<IndexT>   v0_indices;

    virtual void do_build(BuildInfo& info) override
    {
        fem = &require<FiniteElementMethod>();
        fem->add_kinetic_reporter(this);
    }

    virtual void do_init() override
    {
        auto geo_slots = world().scene().geometries();

        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        fem->for_each(geo_slots,
                      [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
                      {
                          auto diff_velocity = sc.vertices().find<Vector3i>("diff/velocity");

                          auto diff_velocity_view = diff_velocity ?
                                                        diff_velocity->view() :
                                                        span<const Vector3i>{};
                          for(auto&& [i, diff_parm_index] : enumerate(diff_velocity_view))
                          {
                              if(diff_parm_index.maxCoeff() >= 0)
                              {
                                  auto vert_offset = I.geo_info().vertex_offset;
                                  h_v0_parm_indices.push_back(diff_parm_index);
                                  h_v0_indices.push_back(vert_offset + i);
                              }
                          }
                      });

        v0_parm_indices.resize(h_v0_parm_indices.size());
        v0_parm_indices.view().copy_from(h_v0_parm_indices.data());

        v0_indices.resize(h_v0_indices.size());
        v0_indices.view().copy_from(h_v0_indices.data());
    }

    virtual void do_report_extent(GlobalDiffSimManager::DiffParmExtentInfo& info) override
    {
        // pGpv0

        auto F = info.frame();

        if(F == 1)  // only at frame 1 we have v0
        {
            // every v0 has 3 dof, so we have 3 matrix3x1 for each v0
            info.triplet_count(h_v0_parm_indices.size() * 3);
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
                    velocities = fem->m_impl.vs.viewer().name("velocities")] __device__(int I) mutable
                   {
                       auto& v0_parm_index = v0_parm_indices(I);
                       auto& v0_index      = v0_indices(I);
                       auto& V0            = velocities(v0_index);

                       for(int i = 0; i < 3; ++i)
                       {
                           auto parm_i = v0_parm_index(i);
                           if(parm_i >= 0)
                           {
                               V0(i) = parameters(parm_i);
                           }
                       }
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
                        masses   = fem->masses().cviewer().name("masses"),
                        is_fixed = fem->is_fixed().cviewer().name("is_fixed"),
                        is_dynamic = fem->is_dynamic().cviewer().name("is_dynamic"),
                        pGpP = info.pGpP().viewer().name("pGpP"),
                        dt   = info.dt()] __device__(int I) mutable
                       {
                           Float    m          = masses(v0_indices(I));
                           auto     vert_index = v0_indices(I);
                           Vector3i parm_index = v0_parm_indices(I);
                           Float    value      = 0.0;

                           if(is_fixed(vert_index))
                           {
                               value = 0.0;
                           }
                           else
                           {
                               value = -m * dt;
                           }

                           Matrix3x3 pGpv0 = value * Matrix3x3::Identity();

                           auto max_parm_index = parm_index.maxCoeff();

                           MUDA_ASSERT(max_parm_index >= 0,
                                       "max_parm_index (%d) < 0, why can it happen?",
                                       max_parm_index);

                           // fill one col of pGpv0 each time
                           for(int i = 0; i < 3; ++i)
                           {
                               auto parm_i = parm_index(i);
                               if(parm_i >= 0)
                               {
                                   pGpP(I * 3 + i).write(
                                       vert_index, parm_i, pGpv0.col(i));
                               }
                               else
                               {
                                   // write to a safe place (max_parm_index) with value 0
                                   pGpP(I * 3 + i).write(vert_index,
                                                         max_parm_index,
                                                         Vector3::Zero());
                               }
                           }
                       });
        }
        else
        {
            UIPC_ASSERT(info.pGpP().triplet_count() == 0, "pGpP size mismatch");
        }
    }
};

REGISTER_SIM_SYSTEM(FiniteElementKineticDiffParmReporter);
}  // namespace uipc::backend::cuda
