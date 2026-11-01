#include <finite_element/codim_2d_constitution.h>
#include <finite_element/constitutions/StVK_shell_2d_function.h>
#include <kernel_cout.h>
#include <muda/ext/eigen/log_proxy.h>
#include <Eigen/Dense>
#include <muda/ext/eigen/inverse.h>
#include <utils/codim_thickness.h>
#include <utils/matrix_assembler.h>
#include <utils/matrix_unpacker.h>
#include <utils/matrix_assembly_utils.h>
#include <finite_element/codim_2d_constitution_diff_parm_reporter.h>

namespace uipc::backend::cuda
{
static constexpr U64 ConstitutionUID = 40;

class StVKShell2D final : public Codim2DConstitution
{
  public:
    // Constitution UID by libuipc specification

    using Codim2DConstitution::Codim2DConstitution;

    vector<Float>   h_k11s, h_k22s, h_k12s, h_k33s;
    vector<Vector6> h_rest_xs;

    muda::DeviceBuffer<Float>   k11s, k22s, k12s, k33s;
    muda::DeviceBuffer<Vector6> rest_xs;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        //std::cout << "In the stvk Initialization" << std::endl;

        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto N = info.primitive_count();

        h_k11s.resize(N);
        h_k22s.resize(N);
        h_k12s.resize(N);
        h_k33s.resize(N);

        k11s.resize(N);
        k22s.resize(N);
        k12s.resize(N);
        k33s.resize(N);

        h_rest_xs.resize(N);
        rest_xs.resize(N);

        list<Float>   k11_list, k12_list, k22_list, k33_list;
        list<Vector6> rest_xs_list;

        info.for_each(geo_slots,
                      [&](geometry::SimplicialComplex& sc) -> auto
                      {
                          auto k11 = sc.triangles().find<Float>("k11");
                          UIPC_ASSERT(k11, "k11 not found, why?");
                          auto k11_view = k11->view();

                          auto k12 = sc.triangles().find<Float>("k12");
                          UIPC_ASSERT(k12, "k12 not found, why?");
                          auto k12_view = k12->view();

                          auto k22 = sc.triangles().find<Float>("k22");
                          UIPC_ASSERT(k22, "k22 not found, why?");
                          auto k22_view = k22->view();

                          auto k33 = sc.triangles().find<Float>("k33");
                          UIPC_ASSERT(k33, "k33 not found, why?");
                          auto k33_view = k33->view();

                          auto rest_x_attr = sc.triangles().find<Vector6>("rest_x");
                          UIPC_ASSERT(rest_x_attr, "rest_x_attr not found, why?");
                          auto rest_x_view = rest_x_attr->view();

                          for(unsigned int tri = 0; tri < sc.triangles().size(); ++tri)
                          {
                              Float k11 = k11_view[tri];
                              k11_list.push_back(k11);

                              Float k12 = k12_view[tri];
                              k12_list.push_back(k12);

                              Float k22 = k22_view[tri];
                              k22_list.push_back(k22);

                              Float k33 = k33_view[tri];
                              k33_list.push_back(k33);

                              Vector6 rest_x = rest_x_view[tri];
                              rest_xs_list.push_back(rest_x);
                          }
                      });

        std::ranges::move(k11_list, h_k11s.begin());
        std::ranges::move(k12_list, h_k12s.begin());
        std::ranges::move(k22_list, h_k22s.begin());
        std::ranges::move(k33_list, h_k33s.begin());

        k11s.view().copy_from(h_k11s.data());
        k12s.view().copy_from(h_k12s.data());
        k22s.view().copy_from(h_k22s.data());
        k33s.view().copy_from(h_k33s.data());

        std::ranges::move(rest_xs_list, h_rest_xs.begin());

        rest_xs.view().copy_from(h_rest_xs.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace STVK = sym::stvk_2d;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [k11s       = k11s.cviewer().name("k11"),
                    k12s       = k12s.cviewer().name("k12"),
                    k22s       = k22s.cviewer().name("k22"),
                    k33s       = k33s.cviewer().name("k33"),
                    rest_xs    = rest_xs.viewer().name("rest_xs"),
                    rest_areas = info.rest_areas().viewer().name("rest_area"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    energies = info.energies().viewer().name("energies"),
                    indices  = info.indices().viewer().name("indices"),
                    xs       = info.xs().viewer().name("xs"),
                    dt       = info.dt()] __device__(int I)
                   {
                       Float k11 = k11s(I);
                       Float k12 = k12s(I);
                       Float k22 = k22s(I);
                       Float k33 = k33s(I);

                       Vector3i idx = indices(I);

                       Vector9 curr_x;

                       curr_x(0) = xs(idx(0))(0);
                       curr_x(1) = xs(idx(0))(1);
                       curr_x(2) = xs(idx(0))(2);

                       curr_x(3) = xs(idx(1))(0);
                       curr_x(4) = xs(idx(1))(1);
                       curr_x(5) = xs(idx(1))(2);

                       curr_x(6) = xs(idx(2))(0);
                       curr_x(7) = xs(idx(2))(1);
                       curr_x(8) = xs(idx(2))(2);

                       Vector6 rest_x = rest_xs(I);

                       Float rest_area = rest_areas(I);
                       Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                            thicknesses(idx(1)),
                                                            thicknesses(idx(2)));

                       Float E;
                       STVK::StVKEnergy(E, k11, k12, k22, k33, curr_x, rest_x);
                       //printf("Energy:{%f}", E);
                       energies(I) = E * rest_area * thickness * dt * dt;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace STVK = sym::stvk_2d;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [k11s    = k11s.cviewer().name("k11"),
                    k12s    = k12s.cviewer().name("k12"),
                    k22s    = k22s.cviewer().name("k22"),
                    k33s    = k33s.cviewer().name("k33"),
                    rest_xs = rest_xs.viewer().name("rest_xs"),
                    indices = info.indices().viewer().name("indices"),
                    xs      = info.xs().viewer().name("xs"),
                    x_bars  = info.x_bars().viewer().name("x_bars"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    G3s        = info.gradients().viewer().name("gradient"),
                    H3x3s      = info.hessians().viewer().name("hessian"),
                    rest_areas = info.rest_areas().viewer().name("volumes"),
                    is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                    dt         = info.dt(),
                    is_adjoint = info.is_adjoint()] __device__(int I) mutable
                   {
                       Float k11 = k11s(I);
                       Float k12 = k12s(I);
                       Float k22 = k22s(I);
                       Float k33 = k33s(I);

                       Vector3i idx = indices(I);

                       Vector9 curr_x;

                       curr_x(0) = xs(idx(0))(0);
                       curr_x(1) = xs(idx(0))(1);
                       curr_x(2) = xs(idx(0))(2);

                       curr_x(3) = xs(idx(1))(0);
                       curr_x(4) = xs(idx(1))(1);
                       curr_x(5) = xs(idx(1))(2);

                       curr_x(6) = xs(idx(2))(0);
                       curr_x(7) = xs(idx(2))(1);
                       curr_x(8) = xs(idx(2))(2);

                       Vector6 rest_x = rest_xs(I);

                       Float rest_area = rest_areas(I);
                       Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                            thicknesses(idx(1)),
                                                            thicknesses(idx(2)));

                       Float Vdt2 = rest_area * thickness * dt * dt;

                       Vector9 grad;
                       STVK::StVKGradient(grad, k11, k12, k22, k33, curr_x, rest_x);
                       grad *= Vdt2;

                       Vector3i ignore = {
                           is_fixed(idx(0)), is_fixed(idx(1)), is_fixed(idx(2))};

                       DoubletVectorAssembler DVA{G3s};
                       DVA.segment<3>(I * 3).write(idx, ignore, grad);

                       Matrix9x9 H;
                       STVK::STVKHessian(H, k11, k12, k22, k33, curr_x, rest_x);
                       H *= Vdt2;
                       // Old: make_spd(H);
                       // Forward Newton still uses the projected Hessian for
                       // stability. The adjoint differentiates the equilibrium
                       // residual and must use its unprojected Jacobian.
                       // Projecting here can bias small, cancelling gradients.
                       if(!is_adjoint)
                           make_spd(H);

                       TripletMatrixAssembler TMA{H3x3s};
                       TMA.block<3, 3>(I * 3 * 3).write(idx, ignore, H);
                   });
    }
};

REGISTER_SIM_SYSTEM(StVKShell2D);

class DiffStVKShell2D : public Codim2DConstitutionDiffParmReporter
{
  public:
    using Base = Codim2DConstitutionDiffParmReporter;
    using Base::Base;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    StVKShell2D& constitution() noexcept
    {
        return static_cast<StVKShell2D&>(Base::constitution());
    }

    vector<IndexT> h_k11_parm_indices;
    vector<IndexT> h_k22_parm_indices;
    vector<IndexT> h_k12_parm_indices;
    vector<IndexT> h_k33_parm_indices;

    vector<IndexT> h_k11_tri_indices;
    vector<IndexT> h_k22_tri_indices;
    vector<IndexT> h_k12_tri_indices;
    vector<IndexT> h_k33_tri_indices;

    muda::DeviceBuffer<IndexT> k11_parm_indices;
    muda::DeviceBuffer<IndexT> k22_parm_indices;
    muda::DeviceBuffer<IndexT> k12_parm_indices;
    muda::DeviceBuffer<IndexT> k33_parm_indices;

    muda::DeviceBuffer<IndexT> k11_tri_indices;
    muda::DeviceBuffer<IndexT> k22_tri_indices;
    muda::DeviceBuffer<IndexT> k12_tri_indices;
    muda::DeviceBuffer<IndexT> k33_tri_indices;

    SizeT k11_pGpP_triplet_count = 0;
    SizeT k22_pGpP_triplet_count = 0;
    SizeT k12_pGpP_triplet_count = 0;
    SizeT k33_pGpP_triplet_count = 0;

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto prim_count = info.primitive_count();

        // k11
        {
            h_k11_parm_indices.reserve(prim_count);
            h_k11_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_k11 = sc.triangles().find<IndexT>("diff/k11");
                    return diff_k11 ? diff_k11->view() : span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_k11)
                {
                    if(diff_k11 >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_k11_parm_indices.push_back(diff_k11);
                        // record the corresponding triangle index
                        h_k11_tri_indices.push_back(I.global_index());
                    }
                });

            h_k11_parm_indices.shrink_to_fit();
            h_k11_tri_indices.shrink_to_fit();

            k11_parm_indices.resize(h_k11_parm_indices.size());
            k11_parm_indices.view().copy_from(h_k11_parm_indices.data());

            k11_tri_indices.resize(h_k11_tri_indices.size());
            k11_tri_indices.view().copy_from(h_k11_tri_indices.data());
        }

        std::cout << "K11 Parameter Size: " << k11_parm_indices.size() << std::endl;

        // k22
        {
            h_k22_parm_indices.reserve(prim_count);
            h_k22_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_k22 = sc.triangles().find<IndexT>("diff/k22");
                    return diff_k22 ? diff_k22->view() : span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_k22)
                {
                    if(diff_k22 >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_k22_parm_indices.push_back(diff_k22);
                        // record the corresponding triangle index
                        h_k22_tri_indices.push_back(I.global_index());
                    }
                });

            h_k22_parm_indices.shrink_to_fit();
            h_k22_tri_indices.shrink_to_fit();

            k22_parm_indices.resize(h_k22_parm_indices.size());
            k22_parm_indices.view().copy_from(h_k22_parm_indices.data());

            k22_tri_indices.resize(h_k22_tri_indices.size());
            k22_tri_indices.view().copy_from(h_k22_tri_indices.data());
        }

        std::cout << "K22 Parameter Size: " << k22_parm_indices.size() << std::endl;

        // k12
        {
            h_k12_parm_indices.reserve(prim_count);
            h_k12_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_k12 = sc.triangles().find<IndexT>("diff/k12");
                    return diff_k12 ? diff_k12->view() : span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_k12)
                {
                    if(diff_k12 >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_k12_parm_indices.push_back(diff_k12);
                        // record the corresponding triangle index
                        h_k12_tri_indices.push_back(I.global_index());
                    }
                });

            h_k12_parm_indices.shrink_to_fit();
            h_k12_tri_indices.shrink_to_fit();

            k12_parm_indices.resize(h_k12_parm_indices.size());
            k12_parm_indices.view().copy_from(h_k12_parm_indices.data());

            k12_tri_indices.resize(h_k12_tri_indices.size());
            k12_tri_indices.view().copy_from(h_k12_tri_indices.data());
        }

        std::cout << "K12 Parameter Size: " << k12_parm_indices.size() << std::endl;

        // k33
        {
            h_k33_parm_indices.reserve(prim_count);
            h_k33_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_k33 = sc.triangles().find<IndexT>("diff/k33");
                    return diff_k33 ? diff_k33->view() : span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_k33)
                {
                    if(diff_k33 >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_k33_parm_indices.push_back(diff_k33);
                        // record the corresponding triangle index
                        h_k33_tri_indices.push_back(I.global_index());
                    }
                });

            h_k33_parm_indices.shrink_to_fit();
            h_k33_tri_indices.shrink_to_fit();

            k33_parm_indices.resize(h_k33_parm_indices.size());
            k33_parm_indices.view().copy_from(h_k33_parm_indices.data());

            k33_tri_indices.resize(h_k33_tri_indices.size());
            k33_tri_indices.view().copy_from(h_k33_tri_indices.data());
        }

        std::cout << "K33 Parameter Size: " << k33_parm_indices.size() << std::endl;

        //std::cout << "Finish the Backward Initialization" << std::endl;
    }

    virtual void do_update(DiffParmUpdateInfo& info) override
    {
        std::cout << "In the Backward Update" << std::endl;
        using namespace muda;

        // update k11
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(k11_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        k11_parm_indices = k11_parm_indices.cviewer().name("k11_parm_indices"),
                        k11_tri_indices = k11_tri_indices.cviewer().name("k11_tri_indices"),
                        k11s = constitution().k11s.viewer().name("k11")] __device__(int I)
                       {
                           // Get the data from parameters and update the k11
                           auto& k11 = k11s(k11_tri_indices(I));
                           k11       = parameters(k11_parm_indices(I));
                       });
        }

        // update k22
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(k22_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        k22_parm_indices = k22_parm_indices.cviewer().name("k22_parm_indices"),
                        k22_tri_indices = k22_tri_indices.cviewer().name("k22_tri_indices"),
                        k22s = constitution().k22s.viewer().name("k22")] __device__(int I)
                       {
                           // Get the data from parameters and update the k22
                           auto& k22 = k22s(k22_tri_indices(I));
                           k22       = parameters(k22_parm_indices(I));
                       });
        }

        // update k12
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(k12_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        k12_parm_indices = k12_parm_indices.cviewer().name("k12_parm_indices"),
                        k12_tri_indices = k12_tri_indices.cviewer().name("k12_tri_indices"),
                        k12s = constitution().k12s.viewer().name("k12")] __device__(int I)
                       {
                           // Get the data from parameters and update the k12
                           auto& k12 = k12s(k12_tri_indices(I));
                           k12       = parameters(k12_parm_indices(I));
                       });
        }

        // update k33
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(k33_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        k33_parm_indices = k33_parm_indices.cviewer().name("k33_parm_indices"),
                        k33_tri_indices = k33_tri_indices.cviewer().name("k33_tri_indices"),
                        k33s = constitution().k33s.viewer().name("k33")] __device__(int I)
                       {
                           // Get the data from parameters and update the k33
                           auto& k33 = k33s(k33_tri_indices(I));
                           k33       = parameters(k33_parm_indices(I));
                       });
        }

        std::cout << "Finish the Backward Update" << std::endl;
    }

    virtual void do_report_extent(DiffParmExtentInfo& info) override
    {
        //std::cout << "In the Backward Report extend" << std::endl;

        // every k11 related to 3 vertices, pGpk11 is 3 * matrix3x1
        {
            auto stencil_dim       = 3;
            auto count             = k11_parm_indices.size();
            k11_pGpP_triplet_count = count * stencil_dim;

            //std::cout << "k11_pGpP_triplet_count " << k11_pGpP_triplet_count << std::endl;
        }

        //std::cout << "k11 pGpP Size: " << k11_pGpP_triplet_count << std::endl;

        // every k22 related to 3 vertices, pGpk22 is 3 * matrix3x1
        {
            auto stencil_dim       = 3;
            auto count             = k22_parm_indices.size();
            k22_pGpP_triplet_count = count * stencil_dim;

            //std::cout << "k22_pGpP_triplet_count " << k22_pGpP_triplet_count << std::endl;
        }

        // every k12 related to 3 vertices, pGpk12 is 3 * matrix3x1
        {
            auto stencil_dim       = 3;
            auto count             = k12_parm_indices.size();
            k12_pGpP_triplet_count = count * stencil_dim;

            //std::cout << "k12_pGpP_triplet_count " << k12_pGpP_triplet_count << std::endl;
        }

        // every k33 related to 3 vertices, pGpk33 is 3 * matrix3x1
        {
            auto stencil_dim       = 3;
            auto count             = k33_parm_indices.size();
            k33_pGpP_triplet_count = count * stencil_dim;

            //std::cout << "k33_pGpP_triplet_count " << k33_pGpP_triplet_count << std::endl;
        }

        info.triplet_count(k11_pGpP_triplet_count + k22_pGpP_triplet_count
                           + k12_pGpP_triplet_count + k33_pGpP_triplet_count);

        //std::cout << "Finish the Backward Report extend" << std::endl;
    }

    virtual void do_assemble(DiffParmInfo& info) override
    {
        //std::cout << "In Do Assemble" << std::endl;

        using namespace muda;
        namespace stvk = sym::stvk_2d;

        auto pGpP = info.pGpP();

        //std::cout << "Assemable Frame: " << info.frame() << std::endl;

        // assemble k11
        {
            auto N = k11_parm_indices.size();
            SizeT k11_offset = 0;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [k11_parm_indices = k11_parm_indices.cviewer().name("k11_parm_indices"),
                     k11_tri_indices = k11_tri_indices.cviewer().name("k11_tri_indices"),
                     pGpP    = pGpP.viewer().name("pGpP"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     x_bars  = info.x_bars().viewer().name("x_bars"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     rest_areas = info.rest_areas().viewer().name("rest_area"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     dt         = info.dt(),
                    k11_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = k11_tri_indices(I);
                        auto parm_idx = k11_parm_indices(I);

                        Vector6 rest_x = rest_xs(tri_idx);

                        Vector3i idx = indices(tri_idx);

                        Float rest_area = rest_areas(tri_idx);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));

                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;
                        auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());


                        Eigen::Matrix<double, 9, 3> Du, Dv;
                        Du.setZero();

                        Du.block<3, 3>(0, 0) =
                            (-dm_inv(0, 0) - dm_inv(1, 0)) * Matrix3x3::Identity();
                        Du.block<3, 3>(3, 0) = dm_inv(0, 0) * Matrix3x3::Identity();
                        Du.block<3, 3>(6, 0) = dm_inv(1, 0) * Matrix3x3::Identity();

                        Vector9 fuu = Du * F.col(0);

                        Float elastic_strain_uu;
                        elastic_strain_uu = G(0, 0);

                        Vector9 pGpk11;
                        pGpk11 = elastic_strain_uu * fuu * Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpk11, ignore);

                        //printf("k11 offset: %d\r\n", k11_offset);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(k11_offset + I * 3 + k).write(vI, J, pGpk11.segment<3>(k * 3));
                        }
                    });
        }

        // assemble k22
        {
            //std::cout << "About to assemable k22" << std::endl;

            auto N = k22_parm_indices.size();
            SizeT k22_offset = k11_pGpP_triplet_count;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [k22_parm_indices = k22_parm_indices.cviewer().name("k22_parm_indices"),
                     k22_tri_indices = k22_tri_indices.cviewer().name("k22_tri_indices"),
                     pGpP    = pGpP.viewer().name("pGpP"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     x_bars  = info.x_bars().viewer().name("x_bars"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     rest_areas = info.rest_areas().viewer().name("rest_area"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     dt         = info.dt(),
                     k22_offset] __device__(int I) mutable
                    {

                        auto tri_idx  = k22_tri_indices(I);
                        auto parm_idx = k22_parm_indices(I);

                        Vector6 rest_x = rest_xs(tri_idx);

                        Vector3i idx = indices(tri_idx);

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;
                        auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());


                        Eigen::Matrix<double, 9, 3> Dv;
                        Dv.setZero();

                        Dv.block<3, 3>(0, 0) =
                            (-dm_inv(0, 1) - dm_inv(1, 1)) * Matrix3x3::Identity();
                        Dv.block<3, 3>(3, 0) = dm_inv(0, 1) * Matrix3x3::Identity();
                        Dv.block<3, 3>(6, 0) = dm_inv(1, 1) * Matrix3x3::Identity();

                        Vector9 fvv = Dv * F.col(1);

                        Float elastic_strain_vv;
                        elastic_strain_vv = G(1, 1);

                        Float rest_area = rest_areas(tri_idx);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));

                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Vector9 pGpk22;
                        pGpk22 = elastic_strain_vv * fvv * Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};


                        zero_out(pGpk22, ignore);

                        //printf("k22 offset: %d\r\n", k22_offset);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(k22_offset + I * 3 + k).write(vI, J, pGpk22.segment<3>(k * 3));
                        }
                    });

            //std::cout << "Finish k22 assemable" << std::endl;
        }

        // assemble k12
        {
            //std::cout << "about to start k12 assemble" << std::endl;

            auto N = k12_parm_indices.size();
            SizeT k12_offset = k11_pGpP_triplet_count + k22_pGpP_triplet_count;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [k12_parm_indices = k12_parm_indices.cviewer().name("k12_parm_indices"),
                     k12_tri_indices = k12_tri_indices.cviewer().name("k12_tri_indices"),
                     k11s    = constitution().k11s.cviewer().name("k11s"),
                     k12s    = constitution().k12s.cviewer().name("k12s"),
                     k22s    = constitution().k11s.cviewer().name("k22s"),
                     k33s    = constitution().k11s.cviewer().name("k33s"),
                     pGpP    = pGpP.viewer().name("pGpP"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     x_bars  = info.x_bars().viewer().name("x_bars"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     rest_areas = info.rest_areas().viewer().name("rest_area"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     dt         = info.dt(),
                     k12_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = k12_tri_indices(I);
                        auto parm_idx = k12_parm_indices(I);

                        Vector6 rest_x = rest_xs(tri_idx);

                        Vector3i idx = indices(tri_idx);

                        //Vector9 curr_x;

                        //curr_x.segment<3>(0) = xs(idx(0));
                        //curr_x.segment<3>(3) = xs(idx(1));
                        //curr_x.segment<3>(6) = xs(idx(2));

                        Float rest_area = rest_areas(tri_idx);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));
                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;
                        auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());

                        Eigen::Matrix<double, 9, 3> Du, Dv;
                        Du.setZero();
                        Dv.setZero();

                        Du.block<3, 3>(0, 0) =
                            (-dm_inv(0, 0) - dm_inv(1, 0)) * Matrix3x3::Identity();
                        Du.block<3, 3>(3, 0) = dm_inv(0, 0) * Matrix3x3::Identity();
                        Du.block<3, 3>(6, 0) = dm_inv(1, 0) * Matrix3x3::Identity();

                        Dv.block<3, 3>(0, 0) =
                            (-dm_inv(0, 1) - dm_inv(1, 1)) * Matrix3x3::Identity();
                        Dv.block<3, 3>(3, 0) = dm_inv(0, 1) * Matrix3x3::Identity();
                        Dv.block<3, 3>(6, 0) = dm_inv(1, 1) * Matrix3x3::Identity();

                        Vector9 fuu = Du * F.col(0);
                        Vector9 fvv = Dv * F.col(1);

                        //Float Vdt2 = rest_area * thickness * dt * dt;

                        Float elastic_strain_uu, elastic_strain_vv;
                        elastic_strain_uu = G(0, 0);
                        elastic_strain_vv = G(1, 1);

                        Vector9 pGpk12;
                        pGpk12 = (elastic_strain_uu * fvv + elastic_strain_vv * fuu) * Vdt2;

                        //Float k11 = k11s(tri_idx);
                        //Float k12 = k12s(tri_idx);
                        //Float k22 = k22s(tri_idx);
                        //Float k33 = k33s(tri_idx);

                        //Vector9 pGpk12;
                        //stvk::dGdk12(pGpk12, k11, k12, k22, k33, curr_x, rest_x);
                        //pGpk12 *= Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpk12, ignore);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(k12_offset + I * 3 + k).write(vI, J, pGpk12.segment<3>(k * 3));
                        }
                    });
        }

        // assemble k33
        {
            auto N = k33_parm_indices.size();
            SizeT k33_offset = k11_pGpP_triplet_count + k22_pGpP_triplet_count + k12_pGpP_triplet_count;

            //std::cout << "Assemable k33" << std::endl;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [k33_parm_indices = k33_parm_indices.cviewer().name("k33_parm_indices"),
                     k33_tri_indices = k33_tri_indices.cviewer().name("k33_tri_indices"),
                     pGpP    = pGpP.viewer().name("pGpP"),
                     rest_xs = constitution().rest_xs.viewer().name("rest_x"),
                     indices = info.indices().viewer().name("indices"),
                     xs      = info.xs().viewer().name("xs"),
                     x_bars  = info.x_bars().viewer().name("x_bars"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     rest_areas = info.rest_areas().viewer().name("rest_area"),
                     is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                     dt         = info.dt(),
                     k33_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = k33_tri_indices(I);
                        auto parm_idx = k33_parm_indices(I);

                        Vector6 rest_x = rest_xs(tri_idx);

                        Vector3i idx = indices(tri_idx);

                        Vector3 a, b, c;

                        a = xs(idx(0));
                        b = xs(idx(1));
                        c = xs(idx(2));

                        Vector2 a_bar, b_bar, c_bar;

                        a_bar = rest_x.segment<2>(0);
                        b_bar = rest_x.segment<2>(2);
                        c_bar = rest_x.segment<2>(4);

                        Matrix2x2 face_dm_m;

                        face_dm_m.col(0) = (b_bar - a_bar);
                        face_dm_m.col(1) = (c_bar - a_bar);

                        Eigen::Matrix<double, 3, 2> face_dm_w;

                        face_dm_w.col(0) = b - a;
                        face_dm_w.col(1) = c - a;

                        auto dm_inv = muda::eigen::inverse(face_dm_m);

                        auto F = face_dm_w * dm_inv;
                        auto G = 0.5 * (F.transpose() * F - Matrix2x2::Identity());

                        Float rest_area = rest_areas(I);
                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));

                        Eigen::Matrix<double, 9, 3> Du, Dv;
                        Du.setZero();
                        Dv.setZero();

                        Du.block<3, 3>(0, 0) =
                            (-dm_inv(0, 0) - dm_inv(1, 0)) * Matrix3x3::Identity();
                        Du.block<3, 3>(3, 0) = dm_inv(0, 0) * Matrix3x3::Identity();
                        Du.block<3, 3>(6, 0) = dm_inv(1, 0) * Matrix3x3::Identity();

                        Dv.block<3, 3>(0, 0) =
                            (-dm_inv(0, 1) - dm_inv(1, 1)) * Matrix3x3::Identity();
                        Dv.block<3, 3>(3, 0) = dm_inv(0, 1) * Matrix3x3::Identity();
                        Dv.block<3, 3>(6, 0) = dm_inv(1, 1) * Matrix3x3::Identity();

                        Vector9 fuv = Du * F.col(1) + Dv * F.col(0);

                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Float elastic_strain_uv;
                        elastic_strain_uv = G(0, 1);

                        Vector9 pGpk33;

                        pGpk33 = elastic_strain_uv * fuv * Vdt2;

                        //pGpk33 = (elastic_strain_uv * fuv
                        //          + 90 * pow(elastic_strain_uv, 2) * fuv)
                        //         * Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpk33, ignore);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(k33_offset + I * 3 + k).write(vI, J, pGpk33.segment<3>(k * 3));
                        }
                    });
        }

        //std::cout << "Finish Do Assemble" << std::endl;
    }
};

REGISTER_SIM_SYSTEM(DiffStVKShell2D);

}  // namespace uipc::backend::cuda


