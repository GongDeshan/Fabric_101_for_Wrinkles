#include <contact_system/simplex_normal_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <utils/distance/distance_flagged.h>
#include <utils/codim_thickness.h>
#include <kernel_cout.h>
#include <contact_system/simplex_normal_contact_diff_parm.h>

namespace uipc::backend::cuda
{
class IPCSimplexNormalContact final : public SimplexNormalContact
{
  public:
    using SimplexNormalContact::SimplexNormalContact;

    virtual void do_build(BuildInfo& info) override
    {
        auto constitution =
            world().scene().info()["contact"]["constitution"].get<std::string>();
        if(constitution != "ipc")
        {
            throw SimSystemException("Constitution is not IPC");
        }
    }

    virtual void do_compute_energy(EnergyInfo& info) override
    {
        using namespace muda;
        using namespace sym::codim_ipc_simplex_contact;

        // Compute Point-Triangle energy
        auto PT_count = info.PTs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(PT_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PTs = info.PTs().viewer().name("PTs"),
                    Es  = info.PT_energies().viewer().name("Es"),
                    Ps  = info.positions().viewer().name("Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       Vector4i PT = PTs(i);

                       Vector4i cids  = {contact_ids(PT[0]),
                                         contact_ids(PT[1]),
                                         contact_ids(PT[2]),
                                         contact_ids(PT[3])};
                       Float    kappa = PT_kappa(table, cids) * dt * dt;

                       const auto& P  = Ps(PT[0]);
                       const auto& T0 = Ps(PT[1]);
                       const auto& T1 = Ps(PT[2]);
                       const auto& T2 = Ps(PT[3]);


                       Float thickness = PT_thickness(thicknesses(PT(0)),
                                                      thicknesses(PT(1)),
                                                      thicknesses(PT(2)),
                                                      thicknesses(PT(3)));

                       Vector4i flag =
                           distance::point_triangle_distance_flag(P, T0, T1, T2);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::point_triangle_distance2(flag, P, T0, T1, T2, D);

                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "PT[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                       PT(0),
                                       PT(1),
                                       PT(2),
                                       PT(3),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       Es(i) = PT_barrier_energy(flag, kappa, d_hat, thickness, P, T0, T1, T2);
                   });

        // Compute Edge-Edge energy
        auto EE_count = info.EEs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(EE_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    EEs = info.EEs().viewer().name("EEs"),
                    Es  = info.EE_energies().viewer().name("Es"),
                    Ps  = info.positions().viewer().name("Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                    d_hat   = info.d_hat(),
                    dt      = info.dt()] __device__(int i) mutable
                   {
                       Vector4i EE = EEs(i);

                       Vector4i cids  = {contact_ids(EE[0]),
                                         contact_ids(EE[1]),
                                         contact_ids(EE[2]),
                                         contact_ids(EE[3])};
                       Float    kappa = EE_kappa(table, cids) * dt * dt;

                       const auto& E0 = Ps(EE[0]);
                       const auto& E1 = Ps(EE[1]);
                       const auto& E2 = Ps(EE[2]);
                       const auto& E3 = Ps(EE[3]);

                       const auto& t0_Ea0 = rest_Ps(EE[0]);
                       const auto& t0_Ea1 = rest_Ps(EE[1]);
                       const auto& t0_Eb0 = rest_Ps(EE[2]);
                       const auto& t0_Eb1 = rest_Ps(EE[3]);

                       Float thickness = EE_thickness(thicknesses(EE(0)),
                                                      thicknesses(EE(1)),
                                                      thicknesses(EE(2)),
                                                      thicknesses(EE(3)));

                       Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::edge_edge_distance2(flag, E0, E1, E2, E3, D);
                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "EE[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f), [%d,%d,%d,%d]",
                                       EE(0),
                                       EE(1),
                                       EE(2),
                                       EE(3),
                                       D,
                                       range(0),
                                       range(1),
                                       flag(0),
                                       flag(1),
                                       flag(2),
                                       flag(3));
                       }


                       Es(i) = mollified_EE_barrier_energy(flag,
                                                           // coefficients
                                                           kappa,
                                                           d_hat,
                                                           thickness,
                                                           // positions
                                                           t0_Ea0,
                                                           t0_Ea1,
                                                           t0_Eb0,
                                                           t0_Eb1,
                                                           E0,
                                                           E1,
                                                           E2,
                                                           E3);
                   });

        // Compute Point-Edge energy
        auto PE_count = info.PEs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(PE_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PEs     = info.PEs().viewer().name("PEs"),
                    Es      = info.PE_energies().viewer().name("Es"),
                    Ps      = info.positions().viewer().name("Ps"),
                    rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    eps_v = info.eps_velocity(),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       Vector3i PE = PEs(i);

                       Vector3i cids  = {contact_ids(PE[0]),
                                         contact_ids(PE[1]),
                                         contact_ids(PE[2])};
                       Float    kappa = PE_kappa(table, cids) * dt * dt;

                       const auto& P  = Ps(PE[0]);
                       const auto& E0 = Ps(PE[1]);
                       const auto& E1 = Ps(PE[2]);

                       Float thickness = PE_thickness(thicknesses(PE(0)),
                                                      thicknesses(PE(1)),
                                                      thicknesses(PE(2)));

                       Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::point_edge_distance2(flag, P, E0, E1, D);

                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "PE[%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                       PE(0),
                                       PE(1),
                                       PE(2),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       Es(i) = PE_barrier_energy(flag, kappa, d_hat, thickness, P, E0, E1);
                   });

        // Compute Point-Point energy
        auto PP_count = info.PPs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(PP_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PPs     = info.PPs().viewer().name("PPs"),
                    Es      = info.PP_energies().viewer().name("Es"),
                    Ps      = info.positions().viewer().name("Ps"),
                    rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       Vector2i PP = PPs(i);

                       Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                       Float    kappa = PP_kappa(table, cids) * dt * dt;

                       const auto& Pa = Ps(PP[0]);
                       const auto& Pb = Ps(PP[1]);

                       Float thickness =
                           PP_thickness(thicknesses(PP(0)), thicknesses(PP(1)));

                       Vector2i flag = distance::point_point_distance_flag(Pa, Pb);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::point_point_distance2(flag, Pa, Pb, D);

                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "PP[%d,%d] d^2(%f) out of range, (%f,%f)",
                                       PP(0),
                                       PP(1),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       Es(i) = PP_barrier_energy(flag, kappa, d_hat, thickness, Pa, Pb);
                   });
    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace muda;
        using namespace sym::codim_ipc_simplex_contact;

        if(info.PTs().size())
        {
            // Compute Point-Triangle Gradient and Hessian
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(info.PTs().size(),
                       [table = info.contact_tabular().viewer().name("contact_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                        PTs = info.PTs().viewer().name("PTs"),
                        Gs  = info.PT_gradients().viewer().name("Gs"),
                        Hs  = info.PT_hessians().viewer().name("Hs"),
                        Ps  = info.positions().viewer().name("Ps"),
                        thicknesses = info.thicknesses().viewer().name("thicknesses"),
                        d_hat = info.d_hat(),
                        dt    = info.dt()] __device__(int i) mutable
                       {
                           Vector4i PT = PTs(i);

                           Vector4i cids  = {contact_ids(PT[0]),
                                             contact_ids(PT[1]),
                                             contact_ids(PT[2]),
                                             contact_ids(PT[3])};
                           Float    kappa = PT_kappa(table, cids) * dt * dt;


                           const auto& P  = Ps(PT[0]);
                           const auto& T0 = Ps(PT[1]);
                           const auto& T1 = Ps(PT[2]);
                           const auto& T2 = Ps(PT[3]);


                           Float thickness = PT_thickness(thicknesses(PT(0)),
                                                          thicknesses(PT(1)),
                                                          thicknesses(PT(2)),
                                                          thicknesses(PT(3)));

                           Vector4i flag =
                               distance::point_triangle_distance_flag(P, T0, T1, T2);

                           if constexpr(RUNTIME_CHECK)
                           {
                               Float D;
                               distance::point_triangle_distance2(flag, P, T0, T1, T2, D);

                               Vector2 range = D_range(thickness, d_hat);

                               MUDA_ASSERT(is_active_D(range, D),
                                           "PT[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                           PT(0),
                                           PT(1),
                                           PT(2),
                                           PT(3),
                                           D,
                                           range(0),
                                           range(1));
                           }

                           PT_barrier_gradient_hessian(
                               Gs(i), Hs(i), flag, kappa, d_hat, thickness, P, T0, T1, T2);

                           //cout << "Gs: " << Gs(i).transpose().eval() << "\n"
                           //     << "Hs: " << Hs(i).transpose().eval() << "\n";
                       });
        }


        // Compute Edge-Edge Gradient and Hessian
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.EEs().size(),
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    EEs = info.EEs().viewer().name("EEs"),
                    Gs  = info.EE_gradients().viewer().name("Gs"),
                    Hs  = info.EE_hessians().viewer().name("Hs"),
                    Ps  = info.positions().viewer().name("Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                    d_hat   = info.d_hat(),
                    dt      = info.dt()] __device__(int i) mutable
                   {
                       Vector4i EE = EEs(i);

                       Vector4i cids  = {contact_ids(EE[0]),
                                         contact_ids(EE[1]),
                                         contact_ids(EE[2]),
                                         contact_ids(EE[3])};
                       Float    kappa = EE_kappa(table, cids) * dt * dt;

                       const auto& E0 = Ps(EE[0]);
                       const auto& E1 = Ps(EE[1]);
                       const auto& E2 = Ps(EE[2]);
                       const auto& E3 = Ps(EE[3]);

                       const auto& t0_Ea0 = rest_Ps(EE[0]);
                       const auto& t0_Ea1 = rest_Ps(EE[1]);
                       const auto& t0_Eb0 = rest_Ps(EE[2]);
                       const auto& t0_Eb1 = rest_Ps(EE[3]);

                       Float thickness = EE_thickness(thicknesses(EE(0)),
                                                      thicknesses(EE(1)),
                                                      thicknesses(EE(2)),
                                                      thicknesses(EE(3)));

                       Vector4i flag = distance::edge_edge_distance_flag(E0, E1, E2, E3);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::edge_edge_distance2(flag, E0, E1, E2, E3, D);

                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "EE[%d,%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                       EE(0),
                                       EE(1),
                                       EE(2),
                                       EE(3),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       mollified_EE_barrier_gradient_hessian(
                           Gs(i), Hs(i), flag, kappa, d_hat, thickness, t0_Ea0, t0_Ea1, t0_Eb0, t0_Eb1, E0, E1, E2, E3);
                   });

        // Compute Point-Edge Gradient and Hessian
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.PEs().size(),
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PEs     = info.PEs().viewer().name("PEs"),
                    Gs      = info.PE_gradients().viewer().name("Gs"),
                    Hs      = info.PE_hessians().viewer().name("Hs"),
                    Ps      = info.positions().viewer().name("Ps"),
                    rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       Vector3i PE = PEs(i);

                       Vector3i cids  = {contact_ids(PE[0]),
                                         contact_ids(PE[1]),
                                         contact_ids(PE[2])};
                       Float    kappa = PE_kappa(table, cids) * dt * dt;

                       const auto& P  = Ps(PE[0]);
                       const auto& E0 = Ps(PE[1]);
                       const auto& E1 = Ps(PE[2]);

                       Float thickness = PE_thickness(thicknesses(PE(0)),
                                                      thicknesses(PE(1)),
                                                      thicknesses(PE(2)));

                       Vector3i flag = distance::point_edge_distance_flag(P, E0, E1);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::point_edge_distance2(flag, P, E0, E1, D);

                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "PE[%d,%d,%d] d^2(%f) out of range, (%f,%f)",
                                       PE(0),
                                       PE(1),
                                       PE(2),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       PE_barrier_gradient_hessian(
                           Gs(i), Hs(i), flag, kappa, d_hat, thickness, P, E0, E1);
                   });

        // Compute Point-Point Gradient and Hessian
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.PPs().size(),
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PPs = info.PPs().viewer().name("PPs"),
                    Gs  = info.PP_gradients().viewer().name("Gs"),
                    Hs  = info.PP_hessians().viewer().name("Hs"),
                    Ps  = info.positions().viewer().name("Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       const auto& PP = PPs(i);

                       Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                       Float    kappa = PP_kappa(table, cids) * dt * dt;

                       const auto& P0 = Ps(PP[0]);
                       const auto& P1 = Ps(PP[1]);

                       Float thickness =
                           PP_thickness(thicknesses(PP(0)), thicknesses(PP(1)));

                       Vector2i flag = distance::point_point_distance_flag(P0, P1);

                       if constexpr(RUNTIME_CHECK)
                       {
                           Float D;
                           distance::point_point_distance2(flag, P0, P1, D);

                           Vector2 range = D_range(thickness, d_hat);

                           MUDA_ASSERT(is_active_D(range, D),
                                       "PP[%d,%d] d^2(%f) out of range, (%f,%f)",
                                       PP(0),
                                       PP(1),
                                       D,
                                       range(0),
                                       range(1));
                       }

                       PP_barrier_gradient_hessian(
                           Gs(i), Hs(i), flag, kappa, d_hat, thickness, P0, P1);
                   });
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexNormalContact);

namespace detail
{
    template <int N>
    __device__ void assemble_pGpP(int                                     I,
                                  muda::TripletMatrixViewer<Float, 3, 1>& pGpP,
                                  const Eigen::Vector<Float, 3 * N>& pGpkappa,
                                  const DiffContactCoeff&            diff_coeff,
                                  const Eigen::Vector<IndexT, N>&    indices)
    {
        constexpr int Span       = N;
        int           offset     = I * Span;
        IndexT        diff_kappa = diff_coeff.diff_kappa;
        for(int v = 0; v < N; ++v)
        {

            Vector3 Value;
            if(diff_kappa < 0)
            {
                Value      = Vector3::Zero();
                // FIX (2026-09-20): preserve the disabled-parameter sentinel.
                // Changing it to 0 makes later vertices emit kappa derivatives
                // into parameter 0. Only remap the column at the zero write.
                // OLD (rollback reference): diff_kappa = 0;
            }
            else
            {
                Value = pGpkappa.segment<3>(v * 3);
            }

            // OLD (rollback reference): write(indices[v], diff_kappa, Value);
            pGpP(offset++).write(indices[v],  // Vid I
                                 diff_kappa < 0 ? 0 : diff_kappa,  // Safe zero column
                                 Value);
        }
    }
}  // namespace detail


class IPCSimplexNormalContactDiffParm final : public SimplexNormalContactDiffParm
{
  public:
    using SimplexNormalContactDiffParm::SimplexNormalContactDiffParm;

    // Inherited via SimplexNormalContactDiffParm
    void do_build(BuildInfo& info) override {}

    void do_init(InitInfo&) override {}

    void do_assemble(DiffParmInfo& info) override
    {
        using namespace muda;
        namespace CIC  = sym::codim_ipc_contact;
        namespace CISC = sym::codim_ipc_simplex_contact;

        auto PT_count = info.friction_PTs().size();
        if(PT_count)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(PT_count,
                       [pGpPs = info.PT_pGpP().viewer().name("pGpPs"),
                        table = info.contact_tabular().viewer().name("table"),
                        diff_table = info.diff_contact_tabular().viewer().name("diff_table"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                        PTs = info.friction_PTs().viewer().name("PTs"),
                        Ps  = info.positions().viewer().name("Ps"),
                        prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                        thicknesses = info.thicknesses().viewer().name("thicknesses"),
                        eps_v = info.eps_velocity(),
                        d_hat = info.d_hat(),
                        dt    = info.dt()] __device__(int I) mutable
                       {
                           const Vector4i& PT = PTs(I);

                           Vector4i cids = {contact_ids(PT[0]),
                                            contact_ids(PT[1]),
                                            contact_ids(PT[2]),
                                            contact_ids(PT[3])};

                           auto coeff = CIC::PT_contact_coeff(table, cids);

                           Float kappa = coeff.kappa;
                           Float mu    = coeff.mu;

                           auto diff_coeff = CIC::PT_diff_contact_coeff(diff_table, cids);

                           IndexT diff_kappa = diff_coeff.diff_kappa;
                           IndexT diff_mu    = diff_coeff.diff_mu;

                           Vector12 Xn;
                           Xn.segment<3>(0) = prev_Ps(PT[0]);
                           Xn.segment<3>(3) = prev_Ps(PT[1]);
                           Xn.segment<3>(6) = prev_Ps(PT[2]);
                           Xn.segment<3>(9) = prev_Ps(PT[3]);

                           Vector12 X;
                           X.segment<3>(0) = Ps(PT[0]);
                           X.segment<3>(3) = Ps(PT[1]);
                           X.segment<3>(6) = Ps(PT[2]);
                           X.segment<3>(9) = Ps(PT[3]);

                           Float thickness = PT_thickness(thicknesses(PT[0]),
                                                          thicknesses(PT[1]),
                                                          thicknesses(PT[2]),
                                                          thicknesses(PT[3]));

                           Float dt2    = dt * dt;
                           Float eps_vh = eps_v * dt;

                           Vector12 pGpkappa;

                           CISC::PT_barrier_pGpkappa(pGpkappa, kappa, X, d_hat, thickness);

                           pGpkappa *= dt2;

                           detail::assemble_pGpP<4>(I, pGpPs, pGpkappa, diff_coeff, PT);
                       });
        }

        auto EE_count = info.friction_EEs().size();
        if(EE_count)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    EE_count,
                    [pGpPs = info.EE_pGpP().viewer().name("pGpPs"),
                     table = info.contact_tabular().viewer().name("table"),
                     diff_table = info.diff_contact_tabular().viewer().name("diff_table"),
                     contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                     EEs     = info.friction_EEs().viewer().name("EEs"),
                     Ps      = info.positions().viewer().name("Ps"),
                     prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                     rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     eps_v = info.eps_velocity(),
                     d_hat = info.d_hat(),
                     dt    = info.dt()] __device__(int I) mutable
                    {
                        const Vector4i& EE = EEs(I);

                        Vector4i cids = {contact_ids(EE[0]),
                                         contact_ids(EE[1]),
                                         contact_ids(EE[2]),
                                         contact_ids(EE[3])};

                        auto coeff = CIC::EE_contact_coeff(table, cids);

                        Float kappa = coeff.kappa;
                        Float mu    = coeff.mu;

                        auto diff_coeff = CIC::EE_diff_contact_coeff(diff_table, cids);

                        IndexT diff_kappa = diff_coeff.diff_kappa;
                        IndexT diff_mu    = diff_coeff.diff_mu;

                        Vector12 Xn;
                        Xn.segment<3>(0) = prev_Ps(EE[0]);
                        Xn.segment<3>(3) = prev_Ps(EE[1]);
                        Xn.segment<3>(6) = prev_Ps(EE[2]);
                        Xn.segment<3>(9) = prev_Ps(EE[3]);

                        Vector12 X;
                        X.segment<3>(0) = Ps(EE[0]);
                        X.segment<3>(3) = Ps(EE[1]);
                        X.segment<3>(6) = Ps(EE[2]);
                        X.segment<3>(9) = Ps(EE[3]);

                        Float thickness = PT_thickness(thicknesses(EE[0]),
                                                       thicknesses(EE[1]),
                                                       thicknesses(EE[2]),
                                                       thicknesses(EE[3]));

                        Float eps_x;

                        distance::edge_edge_mollifier_threshold<Float>(
                            rest_Ps(EE[0]), rest_Ps(EE[1]), rest_Ps(EE[2]), rest_Ps(EE[3]), eps_x);

                        Vector12 pGpkappa;

                        if(distance::need_mollify<Float>(Xn.segment<3>(0),
                                                         Xn.segment<3>(3),
                                                         Xn.segment<3>(6),
                                                         Xn.segment<3>(9),
                                                         eps_x))
                        {
                            pGpkappa.setZero();
                        }
                        else
                        {
                            CISC::EE_barrier_pGpkappa(pGpkappa, kappa, X, d_hat, thickness);

                            auto dt2 = dt * dt;

                            pGpkappa *= dt2;
                        }

                        detail::assemble_pGpP<4>(I, pGpPs, pGpkappa, diff_coeff, EE);
                    });
        }

        auto PE_count = info.friction_PEs().size();
        if(PE_count)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(PE_count,
                       [pGpPs = info.PE_pGpP().viewer().name("pGpPs"),
                        table = info.contact_tabular().viewer().name("table"),
                        diff_table = info.diff_contact_tabular().viewer().name("diff_table"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                        PEs = info.friction_PEs().viewer().name("PEs"),
                        Ps  = info.positions().viewer().name("Ps"),
                        prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                        thicknesses = info.thicknesses().viewer().name("thicknesses"),
                        eps_v = info.eps_velocity(),
                        d_hat = info.d_hat(),
                        dt    = info.dt()] __device__(int I) mutable
                       {
                           const Vector3i& PE = PEs(I);

                           Vector3i cids = {contact_ids(PE[0]),
                                            contact_ids(PE[1]),
                                            contact_ids(PE[2])};

                           auto coeff = CIC::PE_contact_coeff(table, cids);

                           Float kappa = coeff.kappa;
                           Float mu    = coeff.mu;

                           auto diff_coeff = CIC::PE_diff_contact_coeff(diff_table, cids);

                           IndexT diff_kappa = diff_coeff.diff_kappa;
                           IndexT diff_mu    = diff_coeff.diff_mu;

                           Vector9 Xn;
                           Xn.segment<3>(0) = prev_Ps(PE[0]);
                           Xn.segment<3>(3) = prev_Ps(PE[1]);
                           Xn.segment<3>(6) = prev_Ps(PE[2]);

                           Vector9 X;
                           X.segment<3>(0) = Ps(PE[0]);
                           X.segment<3>(3) = Ps(PE[1]);
                           X.segment<3>(6) = Ps(PE[2]);

                           Float thickness = PE_thickness(thicknesses(PE[0]),
                                                          thicknesses(PE[1]),
                                                          thicknesses(PE[2]));

                           Float dt2 = dt * dt;

                           Vector9 pGpkappa;
                           CISC::PE_barrier_pGpkappa(pGpkappa, kappa, X, d_hat, thickness);
                           pGpkappa *= dt2;

                           detail::assemble_pGpP<3>(I, pGpPs, pGpkappa, diff_coeff, PE);
                       });
        }

        auto PP_count = info.friction_PPs().size();
        if(PP_count)
        {
            ParallelFor().apply(
                PP_count,
                [pGpPs = info.PP_pGpP().viewer().name("pGpPs"),
                 table = info.contact_tabular().viewer().name("table"),
                 diff_table = info.diff_contact_tabular().viewer().name("diff_table"),
                 contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                 PPs         = info.friction_PPs().viewer().name("PPs"),
                 Ps          = info.positions().viewer().name("Ps"),
                 prev_Ps     = info.prev_positions().viewer().name("prev_Ps"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 eps_v       = info.eps_velocity(),
                 d_hat       = info.d_hat(),
                 dt          = info.dt()] __device__(int I) mutable
                {
                    const Vector2i& PP = PPs(I);

                    Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};

                    auto coeff = CIC::PP_contact_coeff(table, cids);

                    Float kappa = coeff.kappa;
                    Float mu    = coeff.mu;

                    auto diff_coeff = CIC::PP_diff_contact_coeff(diff_table, cids);

                    IndexT diff_kappa = diff_coeff.diff_kappa;
                    IndexT diff_mu    = diff_coeff.diff_mu;

                    Vector6 Xn;
                    Xn.segment<3>(0) = prev_Ps(PP[0]);
                    Xn.segment<3>(3) = prev_Ps(PP[1]);

                    Vector6 X;
                    X.segment<3>(0) = Ps(PP[0]);
                    X.segment<3>(3) = Ps(PP[1]);

                    Float thickness =
                        PP_thickness(thicknesses(PP[0]), thicknesses(PP[1]));

                    Float dt2    = dt * dt;
                    Float eps_vh = eps_v * dt;

                    Vector6 pGpkappa;
                    CISC::PP_barrier_pGpkappa(pGpkappa, kappa, X, d_hat, thickness);
                    pGpkappa *= dt2;

                    detail::assemble_pGpP<2>(I, pGpPs, pGpkappa, diff_coeff, PP);
                });
        }
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexNormalContactDiffParm);
}  // namespace uipc::backend::cuda