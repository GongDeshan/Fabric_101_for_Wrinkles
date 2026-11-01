#include <finite_element/codim_2d_constitution.h>
#include <finite_element/constitutions/neo_hookean_shell_2d_function.h>
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
static constexpr U64 ConstitutionUID = 11;

class NeoHookeanShell2D final : public Codim2DConstitution
{
  public:
    // Constitution UID by libuipc specification

    using Codim2DConstitution::Codim2DConstitution;

    vector<Float> h_mus;
    vector<Float> h_lambdas;

    muda::DeviceBuffer<Float> mus;
    muda::DeviceBuffer<Float> lambdas;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto N = info.primitive_count();

        h_mus.resize(N);
        h_lambdas.resize(N);

        info.for_each(
            geo_slots,
            [](geometry::SimplicialComplex& sc) -> auto
            {
                auto mu     = sc.triangles().find<Float>("mu");
                auto lambda = sc.triangles().find<Float>("lambda");

                return zip(mu->view(), lambda->view());
            },
            [&](const ForEachInfo& I, auto mu_and_lambda)
            {
                auto vI = I.global_index();

                auto&& [mu, lambda] = mu_and_lambda;
                h_mus[vI]           = mu;
                h_lambdas[vI]       = lambda;
            });

        mus.resize(N);
        mus.view().copy_from(h_mus.data());

        lambdas.resize(N);
        lambdas.view().copy_from(h_lambdas.data());
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace NH = sym::shell_neo_hookean_2d;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [mus        = mus.cviewer().name("mus"),
                    lambdas    = lambdas.cviewer().name("lambdas"),
                    rest_areas = info.rest_areas().viewer().name("rest_area"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    energies = info.energies().viewer().name("energies"),
                    indices  = info.indices().viewer().name("indices"),
                    xs       = info.xs().viewer().name("xs"),
                    x_bars   = info.x_bars().viewer().name("x_bars"),
                    dt       = info.dt()] __device__(int I)
                   {
                       Vector9  X;
                       Vector3i idx = indices(I);
                       for(int i = 0; i < 3; ++i)
                           X.segment<3>(3 * i) = xs(idx(i));

                       Vector9 X_bar;
                       for(int i = 0; i < 3; ++i)
                           X_bar.segment<3>(3 * i) = x_bars(idx(i));

                       Matrix2x2 IB;
                       NH::A(IB, X_bar);
                       IB = muda::eigen::inverse(IB);

                       Float mu        = mus(I);
                       Float lambda    = lambdas(I);
                       Float rest_area = rest_areas(I);
                       Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                            thicknesses(idx(1)),
                                                            thicknesses(idx(2)));

                       Float E;
                       NH::E(E, mu, lambda, X, IB);
                       //printf("Energy {%f}\r\n", E);
                       energies(I) = E * rest_area * thickness * dt * dt;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace NH = sym::shell_neo_hookean_2d;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.indices().size(),
                   [mus     = mus.cviewer().name("mus"),
                    lambdas = lambdas.cviewer().name("lambdas"),
                    indices = info.indices().viewer().name("indices"),
                    xs      = info.xs().viewer().name("xs"),
                    x_bars  = info.x_bars().viewer().name("x_bars"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    G3s        = info.gradients().viewer().name("gradient"),
                    H3x3s      = info.hessians().viewer().name("hessian"),
                    rest_areas = info.rest_areas().viewer().name("volumes"),
                    is_fixed   = info.is_fixed().viewer().name("is_fixed"),
                    dt         = info.dt()] __device__(int I) mutable
                   {
                       Vector9  X;
                       Vector3i idx = indices(I);
                       for(int i = 0; i < 3; ++i)
                           X.segment<3>(3 * i) = xs(idx(i));

                       Vector9 X_bar;
                       for(int i = 0; i < 3; ++i)
                           X_bar.segment<3>(3 * i) = x_bars(idx(i));

                       Matrix2x2 IB;
                       NH::A(IB, X_bar);
                       IB = muda::eigen::inverse(IB);

                       Float mu        = mus(I);
                       Float lambda    = lambdas(I);
                       Float rest_area = rest_areas(I);
                       Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                            thicknesses(idx(1)),
                                                            thicknesses(idx(2)));

                       Float Vdt2 = rest_area * thickness * dt * dt;

                       Vector3i ignore = {
                           is_fixed(idx(0)), is_fixed(idx(1)), is_fixed(idx(2))};

                       Vector9 G;
                       NH::dEdX(G, mu, lambda, X, IB);

                       G *= Vdt2;
                       DoubletVectorAssembler DVA{G3s};
                       DVA.segment<3>(I * 3).write(idx, ignore, G);

                       Matrix9x9 H;
                       NH::ddEddX(H, mu, lambda, X, IB);
                       H *= Vdt2;
                       make_spd(H);

                       TripletMatrixAssembler TMA{H3x3s};
                       TMA.block<3, 3>(I * 3 * 3).write(idx, ignore, H);
                   });
    }
};

REGISTER_SIM_SYSTEM(NeoHookeanShell2D);

class DiffNeoHookeanShell2D : public Codim2DConstitutionDiffParmReporter
{
  public:
    using Base = Codim2DConstitutionDiffParmReporter;
    using Base::Base;

    virtual U64 get_uid() const noexcept override { return ConstitutionUID; }

    virtual void do_build(BuildInfo& info) override {}

    NeoHookeanShell2D& constitution() noexcept
    {
        return static_cast<NeoHookeanShell2D&>(Base::constitution());
    }

    vector<IndexT> h_mu_parm_indices;
    vector<IndexT> h_lambda_parm_indices;
    vector<IndexT> h_mu_tri_indices;
    vector<IndexT> h_lambda_tri_indices;

    muda::DeviceBuffer<IndexT> mu_parm_indices;
    muda::DeviceBuffer<IndexT> mu_tri_indices;

    muda::DeviceBuffer<IndexT> lambda_parm_indices;
    muda::DeviceBuffer<IndexT> lambda_tri_indices;

    SizeT mu_pGpP_triplet_count     = 0;
    SizeT lambda_pGpP_triplet_count = 0;

    virtual void do_init(FiniteElementMethod::FilteredInfo& info) override
    {
        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        auto prim_count = info.primitive_count();

        // mu
        {
            h_mu_parm_indices.reserve(prim_count);
            h_mu_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_mu = sc.triangles().find<IndexT>("diff/mu");
                    return diff_mu ? diff_mu->view() : span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_mu)
                {
                    if(diff_mu >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_mu_parm_indices.push_back(diff_mu);
                        // record the corresponding triangle index
                        h_mu_tri_indices.push_back(I.global_index());
                    }
                });

            h_mu_parm_indices.shrink_to_fit();
            h_mu_tri_indices.shrink_to_fit();

            mu_parm_indices.resize(h_mu_parm_indices.size());
            mu_parm_indices.view().copy_from(h_mu_parm_indices.data());

            mu_tri_indices.resize(h_mu_tri_indices.size());
            mu_tri_indices.view().copy_from(h_mu_tri_indices.data());
        }

        // lambda
        {
            h_lambda_parm_indices.reserve(prim_count);
            h_lambda_tri_indices.reserve(prim_count);

            info.for_each(
                geo_slots,
                [](geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_lambda = sc.triangles().find<IndexT>("diff/lambda");
                    return diff_lambda ? diff_lambda->view() : span<const IndexT>{};
                },
                [&](const ForEachInfo& I, IndexT diff_lambda)
                {
                    if(diff_lambda >= 0)
                    {
                        // record the index in diff-sim parameters
                        h_lambda_parm_indices.push_back(diff_lambda);
                        // record the corresponding triangle index
                        h_lambda_tri_indices.push_back(I.global_index());
                    }
                });

            h_lambda_parm_indices.shrink_to_fit();
            h_lambda_tri_indices.shrink_to_fit();

            lambda_parm_indices.resize(h_lambda_parm_indices.size());
            lambda_parm_indices.view().copy_from(h_lambda_parm_indices.data());

            lambda_tri_indices.resize(h_lambda_tri_indices.size());
            lambda_tri_indices.view().copy_from(h_lambda_tri_indices.data());
        }
    }

    virtual void do_update(DiffParmUpdateInfo& info) override
    {
        using namespace muda;

        // update mu
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(mu_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        mu_parm_indices = mu_parm_indices.cviewer().name("mu_parm_indices"),
                        mu_tri_indices = mu_tri_indices.cviewer().name("mu_tri_indices"),
                        mus = constitution().mus.viewer().name("mus")] __device__(int I)
                       {
                           // Get the data from parameters and update the mu
                           auto& mu = mus(mu_tri_indices(I));
                           mu       = parameters(mu_parm_indices(I));
                       });
        }

        // update lambda
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(lambda_tri_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        lambda_parm_indices = lambda_parm_indices.cviewer().name("lambda_parm_indices"),
                        lambda_tri_indices = lambda_tri_indices.cviewer().name("lambda_tri_indices"),
                        lambdas = constitution().lambdas.viewer().name("lambdas")] __device__(int I)
                       {
                           // Get the data from parameters and update the lambda
                           auto& lambda = lambdas(lambda_tri_indices(I));
                           lambda       = parameters(lambda_parm_indices(I));
                       });
        }
    }

    virtual void do_report_extent(DiffParmExtentInfo& info) override
    {
        // every mu related to 3 vertices, pGpmu is 3 * matrix3x1
        {
            auto stencil_dim      = 3;
            auto count            = mu_parm_indices.size();
            mu_pGpP_triplet_count = count * stencil_dim;
        }

        // every lambda related to 3 vertices, pGplambda is 3 * matrix3x1
        {
            auto stencil_dim          = 3;
            auto count                = lambda_parm_indices.size();
            lambda_pGpP_triplet_count = count * stencil_dim;
        }

        info.triplet_count(mu_pGpP_triplet_count + lambda_pGpP_triplet_count);
    }

    virtual void do_assemble(DiffParmInfo& info) override
    {
        using namespace muda;
        namespace NHS = sym::shell_neo_hookean_2d;

        auto pGpP = info.pGpP();

        // assemble mu
        {
            auto  N         = mu_parm_indices.size();
            SizeT mu_offset = 0;
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    N,
                    [mu_parm_indices = mu_parm_indices.cviewer().name("mu_parm_indices"),
                     mu_tri_indices = mu_tri_indices.cviewer().name("mu_tri_indices"),
                     mus     = constitution().mus.cviewer().name("mus"),
                     lambdas = constitution().lambdas.viewer().name("lambdas"),
                     pGpP    = pGpP.viewer().name("pGpP"),
                     rest_areas = info.rest_areas().viewer().name("rest_areas"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     indices  = info.indices().viewer().name("indices"),
                     xs       = info.xs().viewer().name("xs"),
                     x_bars   = info.x_bars().viewer().name("x_bars"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     dt       = info.dt(),
                     mu_offset] __device__(int I) mutable
                    {
                        auto tri_idx  = mu_tri_indices(I);
                        auto parm_idx = mu_parm_indices(I);


                        Vector9  X;
                        Vector3i idx    = indices(tri_idx);
                        X.segment<3>(0) = xs(idx(0));
                        X.segment<3>(3) = xs(idx(1));
                        X.segment<3>(6) = xs(idx(2));

                        Vector9 X_bar;
                        X_bar.segment<3>(0) = x_bars(idx(0));
                        X_bar.segment<3>(3) = x_bars(idx(1));
                        X_bar.segment<3>(6) = x_bars(idx(2));

                        Matrix2x2 IB;
                        NHS::A(IB, X_bar);

                        IB = muda::eigen::inverse(IB);

                        Float rest_area = rest_areas(tri_idx);

                        Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                             thicknesses(idx(1)),
                                                             thicknesses(idx(2)));

                        Float mu     = mus(tri_idx);
                        Float lambda = lambdas(tri_idx);

                        Float Vdt2 = rest_area * thickness * dt * dt;

                        Vector9 pGpmu;
                        NHS::pGpmu(pGpmu, mu, lambda, X, IB);
                        pGpmu *= Vdt2;

                        Vector3i ignore = {is_fixed(idx(0)),
                                           is_fixed(idx(1)),
                                           is_fixed(idx(2))};

                        zero_out(pGpmu, ignore);

                        for(int k = 0; k < 3; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_idx;

                            pGpP(mu_offset + I * 3 + k).write(vI, J, pGpmu.segment<3>(k * 3));
                        }
                    });
        }

        // assemble lambda
        {
            auto  N             = lambda_parm_indices.size();
            SizeT lambda_offset = mu_pGpP_triplet_count;
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(N,
                       [lambda_parm_indices = lambda_parm_indices.cviewer().name("lambda_parm_indices"),
                        lambda_tri_indices = lambda_tri_indices.cviewer().name("lambda_tri_indices"),
                        mus = constitution().mus.viewer().name("mus"),
                        lambdas = constitution().lambdas.cviewer().name("lambdas"),
                        pGpP = pGpP.viewer().name("pGpP"),
                        rest_areas = info.rest_areas().viewer().name("rest_areas"),
                        thicknesses = info.thicknesses().viewer().name("thicknesses"),
                        indices  = info.indices().viewer().name("indices"),
                        xs       = info.xs().viewer().name("xs"),
                        x_bars   = info.x_bars().viewer().name("x_bars"),
                        is_fixed = info.is_fixed().viewer().name("is_fixed"),
                        dt       = info.dt(),
                        lambda_offset] __device__(int I) mutable
                       {
                           auto tri_idx  = lambda_tri_indices(I);
                           auto parm_idx = lambda_parm_indices(I);

                           Vector9  X;
                           Vector3i idx    = indices(tri_idx);
                           X.segment<3>(0) = xs(idx(0));
                           X.segment<3>(3) = xs(idx(1));
                           X.segment<3>(6) = xs(idx(2));

                           Vector9 X_bar;
                           X_bar.segment<3>(0) = x_bars(idx(0));
                           X_bar.segment<3>(3) = x_bars(idx(1));
                           X_bar.segment<3>(6) = x_bars(idx(2));

                           Matrix2x2 IB;
                           NHS::A(IB, X_bar);

                           IB = muda::eigen::inverse(IB);

                           Float rest_area = rest_areas(tri_idx);

                           Float thickness = triangle_thickness(thicknesses(idx(0)),
                                                                thicknesses(idx(1)),
                                                                thicknesses(idx(2)));

                           Float mu     = mus(tri_idx);
                           Float lambda = lambdas(tri_idx);

                           Float Vdt2 = rest_area * thickness * dt * dt;

                           Vector9 pGplambda;

                           NHS::pGplmbd(pGplambda, mu, lambda, X, IB);

                           pGplambda *= Vdt2;

                           Vector3i ignore = {is_fixed(idx(0)),
                                              is_fixed(idx(1)),
                                              is_fixed(idx(2))};

                           zero_out(pGplambda, ignore);

                           for(int k = 0; k < 3; ++k)  // for each vertex
                           {
                               auto vI = idx(k);
                               auto J  = parm_idx;

                               pGpP(lambda_offset + I * 3 + k)
                                   .write(vI, J, pGplambda.segment<3>(k * 3));
                           }
                       });
        }
    }
};

REGISTER_SIM_SYSTEM(DiffNeoHookeanShell2D);
}  // namespace uipc::backend::cuda
