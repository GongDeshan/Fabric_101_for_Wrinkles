#include <contact_system/simplex_frictional_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_frictional_contact_function.h>
#include <utils/codim_thickness.h>
#include <kernel_cout.h>
#include <contact_system/simplex_frictional_contact_diff_dof.h>
#include <contact_system/simplex_frictional_contact_diff_parm.h>
#include <contact_system/contact_models/codim_ipc_simplex_diff_friction_function.h>
#include <utils/matrix_assembler.h>

namespace uipc::backend::cuda
{
class IPCSimplexFrictionalContact final : public SimplexFrictionalContact
{
  public:
    using SimplexFrictionalContact::SimplexFrictionalContact;

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
        using namespace sym::codim_ipc_contact;

        // Compute Point-Triangle energy
        auto PT_count = info.friction_PTs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(PT_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PTs     = info.friction_PTs().viewer().name("PTs"),
                    Es      = info.friction_PT_energies().viewer().name("Es"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    eps_v = info.eps_velocity(),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       const auto& PT = PTs(i);

                       Vector4i cids = {contact_ids(PT[0]),
                                        contact_ids(PT[1]),
                                        contact_ids(PT[2]),
                                        contact_ids(PT[3])};

                       auto  coeff = PT_contact_coeff(table, cids);
                       Float kappa = coeff.kappa;
                       Float mu    = coeff.mu;

                       const auto& prev_P  = prev_Ps(PT[0]);
                       const auto& prev_T0 = prev_Ps(PT[1]);
                       const auto& prev_T1 = prev_Ps(PT[2]);
                       const auto& prev_T2 = prev_Ps(PT[3]);

                       const auto& P  = Ps(PT[0]);
                       const auto& T0 = Ps(PT[1]);
                       const auto& T1 = Ps(PT[2]);
                       const auto& T2 = Ps(PT[3]);


                       Float thickness = PT_thickness(thicknesses(PT[0]),
                                                      thicknesses(PT[1]),
                                                      thicknesses(PT[2]),
                                                      thicknesses(PT[3]));

                       Float dt2 = dt * dt;

                       Es(i) = dt2
                               * PT_friction_energy(kappa,
                                                    d_hat,
                                                    thickness,
                                                    mu,
                                                    eps_v * dt,
                                                    // previous positions
                                                    prev_P,
                                                    prev_T0,
                                                    prev_T1,
                                                    prev_T2,
                                                    // current positions
                                                    P,
                                                    T0,
                                                    T1,
                                                    T2);
                   });

        // Compute Edge-Edge energy
        auto EE_count = info.friction_EEs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(EE_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    EEs     = info.friction_EEs().viewer().name("EEs"),
                    Es      = info.friction_EE_energies().viewer().name("Es"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                    eps_v   = info.eps_velocity(),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       const auto& EE = EEs(i);

                       Vector4i cids = {contact_ids(EE[0]),
                                        contact_ids(EE[1]),
                                        contact_ids(EE[2]),
                                        contact_ids(EE[3])};

                       auto  coeff = EE_contact_coeff(table, cids);
                       Float kappa = coeff.kappa;
                       Float mu    = coeff.mu;

                       const Vector3& rest_Ea0 = rest_Ps(EE[0]);
                       const Vector3& rest_Ea1 = rest_Ps(EE[1]);
                       const Vector3& rest_Eb0 = rest_Ps(EE[2]);
                       const Vector3& rest_Eb1 = rest_Ps(EE[3]);

                       const Vector3& prev_Ea0 = prev_Ps(EE[0]);
                       const Vector3& prev_Ea1 = prev_Ps(EE[1]);
                       const Vector3& prev_Eb0 = prev_Ps(EE[2]);
                       const Vector3& prev_Eb1 = prev_Ps(EE[3]);

                       const Vector3& Ea0 = Ps(EE[0]);
                       const Vector3& Ea1 = Ps(EE[1]);
                       const Vector3& Eb0 = Ps(EE[2]);
                       const Vector3& Eb1 = Ps(EE[3]);

                       Float thickness = EE_thickness(thicknesses(EE[0]),
                                                      thicknesses(EE[1]),
                                                      thicknesses(EE[2]),
                                                      thicknesses(EE[3]));

                       Float eps_x;
                       distance::edge_edge_mollifier_threshold(
                           rest_Ea0, rest_Ea1, rest_Eb0, rest_Eb1, eps_x);
                       if(distance::need_mollify(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, eps_x))
                       // almost parallel, don't compute energy
                       {
                           Es(i) = 0;
                       }
                       else
                       {
                           Float dt2 = dt * dt;
                           Es(i)     = dt2
                                   * EE_friction_energy(kappa,
                                                        d_hat,
                                                        thickness,
                                                        mu,
                                                        eps_v * dt,
                                                        // previous positions
                                                        prev_Ea0,
                                                        prev_Ea1,
                                                        prev_Eb0,
                                                        prev_Eb1,
                                                        // current positions
                                                        Ea0,
                                                        Ea1,
                                                        Eb0,
                                                        Eb1);
                       }
                   });

        // Compute Point-Edge energy
        auto PE_count = info.friction_PEs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(PE_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PEs     = info.friction_PEs().viewer().name("PEs"),
                    Es      = info.friction_PE_energies().viewer().name("Es"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    eps_v = info.eps_velocity(),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       const auto& PE = PEs(i);

                       Vector3i cids = {contact_ids(PE[0]),
                                        contact_ids(PE[1]),
                                        contact_ids(PE[2])};

                       auto  coeff = PE_contact_coeff(table, cids);
                       Float kappa = coeff.kappa;
                       Float mu    = coeff.mu;

                       const Vector3& prev_P  = prev_Ps(PE[0]);
                       const Vector3& prev_E0 = prev_Ps(PE[1]);
                       const Vector3& prev_E1 = prev_Ps(PE[2]);

                       const Vector3& P  = Ps(PE[0]);
                       const Vector3& E0 = Ps(PE[1]);
                       const Vector3& E1 = Ps(PE[2]);

                       Float thickness = PE_thickness(thicknesses(PE[0]),
                                                      thicknesses(PE[1]),
                                                      thicknesses(PE[2]));

                       Float dt2 = dt * dt;

                       Es(i) = dt2
                               * PE_friction_energy(kappa,
                                                    d_hat,
                                                    thickness,
                                                    mu,
                                                    eps_v * dt,
                                                    // previous positions
                                                    prev_P,
                                                    prev_E0,
                                                    prev_E1,
                                                    // current positions
                                                    P,
                                                    E0,
                                                    E1);
                   });

        // Compute Point-Point energy
        auto PP_count = info.friction_PPs().size();
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(PP_count,
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PPs     = info.friction_PPs().viewer().name("PPs"),
                    Es      = info.friction_PP_energies().viewer().name("Es"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    eps_v = info.eps_velocity(),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       const auto& PP = PPs(i);

                       Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                       auto     coeff = PP_contact_coeff(table, cids);
                       Float    kappa = coeff.kappa;
                       Float    mu    = coeff.mu;

                       const Vector3& prev_P0 = prev_Ps(PP[0]);
                       const Vector3& prev_P1 = prev_Ps(PP[1]);

                       const Vector3& P0 = Ps(PP[0]);
                       const Vector3& P1 = Ps(PP[1]);

                       Float thickness =
                           PP_thickness(thicknesses(PP[0]), thicknesses(PP[1]));

                       Float dt2 = dt * dt;

                       Es(i) = dt2
                               * PP_friction_energy(kappa,
                                                    d_hat,
                                                    thickness,
                                                    mu,
                                                    eps_v * dt,
                                                    // previous positions
                                                    prev_P0,
                                                    prev_P1,
                                                    // current positions
                                                    P0,
                                                    P1);
                   });
    }

    virtual void do_assemble(ContactInfo& info) override
    {
        using namespace muda;
        using namespace sym::codim_ipc_contact;

        // Compute Point-Triangle Gradient and Hessian
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.friction_PTs().size(),
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PTs     = info.friction_PTs().viewer().name("PTs"),
                    Gs      = info.friction_PT_gradients().viewer().name("Gs"),
                    Hs      = info.friction_PT_hessians().viewer().name("Hs"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    eps_v = info.eps_velocity(),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       const auto& PT = PTs(i);

                       Vector4i cids = {contact_ids(PT[0]),
                                        contact_ids(PT[1]),
                                        contact_ids(PT[2]),
                                        contact_ids(PT[3])};

                       auto  coeff = PT_contact_coeff(table, cids);
                       Float kappa = coeff.kappa;
                       Float mu    = coeff.mu;

                       const auto& prev_P  = prev_Ps(PT[0]);
                       const auto& prev_T0 = prev_Ps(PT[1]);
                       const auto& prev_T1 = prev_Ps(PT[2]);
                       const auto& prev_T2 = prev_Ps(PT[3]);

                       const auto& P  = Ps(PT[0]);
                       const auto& T0 = Ps(PT[1]);
                       const auto& T1 = Ps(PT[2]);
                       const auto& T2 = Ps(PT[3]);


                       Float thickness = PT_thickness(thicknesses(PT[0]),
                                                      thicknesses(PT[1]),
                                                      thicknesses(PT[2]),
                                                      thicknesses(PT[3]));

                       Float dt2 = dt * dt;

                       Vector12    G;
                       Matrix12x12 H;
                       PT_friction_gradient_hessian(G,
                                                    H,
                                                    kappa,
                                                    d_hat,
                                                    thickness,
                                                    mu,
                                                    eps_v * dt,
                                                    // previous positions
                                                    prev_P,
                                                    prev_T0,
                                                    prev_T1,
                                                    prev_T2,
                                                    // current positions
                                                    P,
                                                    T0,
                                                    T1,
                                                    T2);

                       Gs(i) = G * dt2;
                       Hs(i) = H * dt2;
                   });

        // Compute Edge-Edge Gradient and Hessian
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.friction_EEs().size(),
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    EEs     = info.friction_EEs().viewer().name("EEs"),
                    Gs      = info.friction_EE_gradients().viewer().name("Gs"),
                    Hs      = info.friction_EE_hessians().viewer().name("Hs"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    eps_v = info.eps_velocity(),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       const auto& EE = EEs(i);

                       Vector4i cids = {contact_ids(EE[0]),
                                        contact_ids(EE[1]),
                                        contact_ids(EE[2]),
                                        contact_ids(EE[3])};

                       auto  coeff = EE_contact_coeff(table, cids);
                       Float kappa = coeff.kappa;
                       Float mu    = coeff.mu;

                       const Vector3& rest_Ea0 = rest_Ps(EE[0]);
                       const Vector3& rest_Ea1 = rest_Ps(EE[1]);
                       const Vector3& rest_Eb0 = rest_Ps(EE[2]);
                       const Vector3& rest_Eb1 = rest_Ps(EE[3]);

                       const Vector3& prev_Ea0 = prev_Ps(EE[0]);
                       const Vector3& prev_Ea1 = prev_Ps(EE[1]);
                       const Vector3& prev_Eb0 = prev_Ps(EE[2]);
                       const Vector3& prev_Eb1 = prev_Ps(EE[3]);

                       const Vector3& Ea0 = Ps(EE[0]);
                       const Vector3& Ea1 = Ps(EE[1]);
                       const Vector3& Eb0 = Ps(EE[2]);
                       const Vector3& Eb1 = Ps(EE[3]);

                       Float thickness = EE_thickness(thicknesses(EE[0]),
                                                      thicknesses(EE[1]),
                                                      thicknesses(EE[2]),
                                                      thicknesses(EE[3]));

                       Float eps_x;
                       distance::edge_edge_mollifier_threshold(
                           rest_Ea0, rest_Ea1, rest_Eb0, rest_Eb1, eps_x);

                       if(distance::need_mollify(prev_Ea0, prev_Ea1, prev_Eb0, prev_Eb1, eps_x))
                       // almost parallel, don't compute gradient and hessian
                       {

                           Gs(i).setZero();
                           Hs(i).setZero();
                       }
                       else
                       {
                           Float dt2 = dt * dt;

                           Vector12    G;
                           Matrix12x12 H;

                           EE_friction_gradient_hessian(G,
                                                        H,
                                                        kappa,
                                                        d_hat,
                                                        thickness,
                                                        mu,
                                                        eps_v * dt,
                                                        // previous positions
                                                        prev_Ea0,
                                                        prev_Ea1,
                                                        prev_Eb0,
                                                        prev_Eb1,
                                                        // current positions
                                                        Ea0,
                                                        Ea1,
                                                        Eb0,
                                                        Eb1);

                           Gs(i) = G * dt2;
                           Hs(i) = H * dt2;
                       }
                   });

        // Compute Point-Edge Gradient and Hessian
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.friction_PEs().size(),
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PEs     = info.friction_PEs().viewer().name("PEs"),
                    Gs      = info.friction_PE_gradients().viewer().name("Gs"),
                    Hs      = info.friction_PE_hessians().viewer().name("Hs"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    eps_v = info.eps_velocity(),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       const auto& PE = PEs(i);

                       Vector3i cids = {contact_ids(PE[0]),
                                        contact_ids(PE[1]),
                                        contact_ids(PE[2])};

                       auto  coeff = PE_contact_coeff(table, cids);
                       Float kappa = coeff.kappa;
                       Float mu    = coeff.mu;

                       const Vector3& prev_P  = prev_Ps(PE[0]);
                       const Vector3& prev_E0 = prev_Ps(PE[1]);
                       const Vector3& prev_E1 = prev_Ps(PE[2]);

                       const Vector3& P  = Ps(PE[0]);
                       const Vector3& E0 = Ps(PE[1]);
                       const Vector3& E1 = Ps(PE[2]);

                       Float thickness = PE_thickness(thicknesses(PE[0]),
                                                      thicknesses(PE[1]),
                                                      thicknesses(PE[2]));

                       Float dt2 = dt * dt;

                       Vector9   G;
                       Matrix9x9 H;

                       PE_friction_gradient_hessian(G,
                                                    H,
                                                    kappa,
                                                    d_hat,
                                                    thickness,
                                                    mu,
                                                    eps_v * dt,
                                                    // previous positions
                                                    prev_P,
                                                    prev_E0,
                                                    prev_E1,
                                                    // current positions
                                                    P,
                                                    E0,
                                                    E1);

                       Gs(i) = G * dt2;
                       Hs(i) = H * dt2;
                   });

        // Compute Point-Point Gradient and Hessian
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.friction_PPs().size(),
                   [table = info.contact_tabular().viewer().name("contact_tabular"),
                    contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                    PPs     = info.friction_PPs().viewer().name("PPs"),
                    Gs      = info.friction_PP_gradients().viewer().name("Gs"),
                    Hs      = info.friction_PP_hessians().viewer().name("Hs"),
                    Ps      = info.positions().viewer().name("Ps"),
                    prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    eps_v = info.eps_velocity(),
                    d_hat = info.d_hat(),
                    dt    = info.dt()] __device__(int i) mutable
                   {
                       const auto& PP = PPs(i);

                       Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};
                       auto     coeff = PP_contact_coeff(table, cids);
                       Float    kappa = coeff.kappa;
                       Float    mu    = coeff.mu;

                       const Vector3& prev_P0 = prev_Ps(PP[0]);
                       const Vector3& prev_P1 = prev_Ps(PP[1]);

                       const Vector3& P0 = Ps(PP[0]);
                       const Vector3& P1 = Ps(PP[1]);

                       Float thickness =
                           PP_thickness(thicknesses(PP[0]), thicknesses(PP[1]));

                       Float dt2 = dt * dt;

                       Vector6   G;
                       Matrix6x6 H;

                       PP_friction_gradient_hessian(G,
                                                    H,
                                                    kappa,
                                                    d_hat,
                                                    thickness,
                                                    mu,
                                                    eps_v * dt,
                                                    // previous positions
                                                    prev_P0,
                                                    prev_P1,
                                                    // current positions
                                                    P0,
                                                    P1);

                       Gs(i) = G * dt2;
                       Hs(i) = H * dt2;
                   });
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexFrictionalContact);

class IPCSimplexFrictionalContactDiffDof final : public SimplexFrictionalContactDiffDof
{
  public:
    using SimplexFrictionalContactDiffDof::SimplexFrictionalContactDiffDof;

    SimSystemSlot<GlobalTrajectoryFilter> global_trajectory_filter;
    SimSystemSlot<GlobalContactManager>   global_contact_manager;
    SimSystemSlot<GlobalVertexManager>    global_vertex_manager;

    virtual void do_build(BuildInfo& info) override {}

    virtual void do_init(InitInfo& info) override {}

    virtual void do_assemble(DiffDofInfo& info) override
    {
        using namespace muda;
        using namespace sym::codim_ipc_contact;
        namespace DF = sym::codim_ipc_diff_friction;

        // 1) PT

        auto PT_count = info.friction_PTs().size();
        if(PT_count)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(PT_count,
                       [H = info.PT_H().viewer().name("H"),
                        table = info.contact_tabular().viewer().name("contact_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                        PTs = info.friction_PTs().viewer().name("PTs"),
                        Ps  = info.positions().viewer().name("Ps"),
                        prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                        thicknesses = info.thicknesses().viewer().name("thicknesses"),
                        eps_v = info.eps_velocity(),
                        d_hat = info.d_hat(),
                        dt    = info.dt()] __device__(int i) mutable
                       {
                           const Vector4i& PT = PTs(i);

                           Vector4i cids = {contact_ids(PT[0]),
                                            contact_ids(PT[1]),
                                            contact_ids(PT[2]),
                                            contact_ids(PT[3])};

                           auto coeff = PT_contact_coeff(table, cids);

                           Float kappa = coeff.kappa;
                           Float mu    = coeff.mu;

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

                           Matrix12x12 pGpXn =
                               DF::PT_pGpXn(kappa, mu, eps_vh, d_hat, thickness, X, Xn);

                           pGpXn *= dt2;

                           TripletMatrixAssembler TMA{H};
                           TMA.block<4, 4>(i * 4 * 4).write(PT, pGpXn);

                           // cout << "PT:pGpXn:\n" << pGpXn << "\n";
                       });
        }


        // 2) EE

        auto EE_count = info.friction_EEs().size();
        if(EE_count)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(EE_count,
                       [H = info.EE_H().viewer().name("H"),
                        table = info.contact_tabular().viewer().name("contact_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                        EEs = info.friction_EEs().viewer().name("EEs"),
                        Ps  = info.positions().viewer().name("Ps"),
                        prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                        rest_Ps = info.rest_positions().viewer().name("rest_Ps"),
                        eps_v = info.eps_velocity(),
                        thicknesses = info.thicknesses().viewer().name("thicknesses"),
                        d_hat = info.d_hat(),
                        dt    = info.dt()] __device__(int i) mutable
                       {
                           const Vector4i& EE = EEs(i);

                           Vector4i cids = {contact_ids(EE[0]),
                                            contact_ids(EE[1]),
                                            contact_ids(EE[2]),
                                            contact_ids(EE[3])};

                           auto coeff = EE_contact_coeff(table, cids);

                           Float kappa = coeff.kappa;
                           Float mu    = coeff.mu;

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

                           Float thickness = EE_thickness(thicknesses(EE[0]),
                                                          thicknesses(EE[1]),
                                                          thicknesses(EE[2]),
                                                          thicknesses(EE[3]));

                           Float eps_x;
                           distance::edge_edge_mollifier_threshold<Float>(
                               rest_Ps(EE[0]), rest_Ps(EE[1]), rest_Ps(EE[2]), rest_Ps(EE[3]), eps_x);

                           Matrix12x12 pGpXn;
                           if(distance::need_mollify<Float>(Xn.segment<3>(0),
                                                            Xn.segment<3>(3),
                                                            Xn.segment<3>(6),
                                                            Xn.segment<3>(9),
                                                            eps_x))
                           {
                               pGpXn.setZero();
                           }
                           else
                           {
                               pGpXn = DF::EE_pGpXn(
                                   kappa, mu, eps_v * dt, d_hat, thickness, X, Xn);

                               // pGpXn.setZero();
                           }

                           pGpXn *= dt * dt;

                           TripletMatrixAssembler TMA{H};
                           TMA.block<4, 4>(i * 4 * 4).write(EE, pGpXn);

                           // cout << "EE:pGpXn:\n" << pGpXn << "\n";
                       });
        }


        // 3) PE

        auto PE_count = info.friction_PEs().size();
        if(PE_count)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(PE_count,
                       [H = info.PE_H().viewer().name("H"),
                        table = info.contact_tabular().viewer().name("contact_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                        PEs = info.friction_PEs().viewer().name("PEs"),
                        Ps  = info.positions().viewer().name("Ps"),
                        prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                        thicknesses = info.thicknesses().viewer().name("thicknesses"),
                        eps_v = info.eps_velocity(),
                        d_hat = info.d_hat(),
                        dt    = info.dt()] __device__(int i) mutable
                       {
                           const Vector3i& PE = PEs(i);

                           Vector3i cids = {contact_ids(PE[0]),
                                            contact_ids(PE[1]),
                                            contact_ids(PE[2])};

                           auto coeff = PE_contact_coeff(table, cids);

                           Float kappa = coeff.kappa;
                           Float mu    = coeff.mu;

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

                           Float dt2    = dt * dt;
                           Float eps_vh = eps_v * dt;

                           Matrix9x9 pGpXn =
                               DF::PE_pGpXn(kappa, mu, eps_vh, d_hat, thickness, X, Xn);

                           pGpXn *= dt2;


                           TripletMatrixAssembler TMA{H};
                           TMA.block<3, 3>(i * 3 * 3).write(PE, pGpXn);


                           // cout << "PE:pGpXn:\n" << pGpXn << "\n";
                       });
        }


        // 4) PP
        auto PP_count = info.friction_PPs().size();
        if(PP_count)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(PP_count,
                       [H = info.PP_H().viewer().name("H"),
                        table = info.contact_tabular().viewer().name("contact_tabular"),
                        contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                        PPs = info.friction_PPs().viewer().name("PPs"),
                        Ps  = info.positions().viewer().name("Ps"),
                        prev_Ps = info.prev_positions().viewer().name("prev_Ps"),
                        thicknesses = info.thicknesses().viewer().name("thicknesses"),
                        eps_v = info.eps_velocity(),
                        d_hat = info.d_hat(),
                        dt    = info.dt()] __device__(int i) mutable
                       {
                           const Vector2i& PP = PPs(i);

                           Vector2i cids = {contact_ids(PP[0]), contact_ids(PP[1])};

                           auto coeff = PP_contact_coeff(table, cids);

                           Float kappa = coeff.kappa;
                           Float mu    = coeff.mu;

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

                           Matrix6x6 pGpXn =
                               DF::PP_pGpXn(kappa, mu, eps_v * dt, d_hat, thickness, X, Xn);

                           pGpXn *= dt2;

                           TripletMatrixAssembler TMA{H};
                           TMA.block<2, 2>(i * 2 * 2).write(PP, pGpXn);

                           // cout << "PP:pGpXn:\n" << pGpXn << "\n";
                       });
        }
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexFrictionalContactDiffDof);

namespace detail
{
    template <int N>
    __device__ void assemble_pGpP(int                                     I,
                                  muda::TripletMatrixViewer<Float, 3, 1>& pGpP,
                                  const Eigen::Vector<Float, 3 * N>& pGpkappa,
                                  const Eigen::Vector<Float, 3 * N>& pGpmu,
                                  const DiffContactCoeff&            diff_coeff,
                                  const Eigen::Vector<IndexT, N>&    indices)
    {
        constexpr int Span       = 2 * N;
        int           offset     = I * Span;
        IndexT        diff_kappa = diff_coeff.diff_kappa;
        IndexT        diff_mu    = diff_coeff.diff_mu;
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

            if(diff_mu < 0)
            {
                Value   = Vector3::Zero();
                // FIX (2026-09-20): same sentinel issue as kappa above.
                // Disabled mu must contribute zero for EVERY vertex.
                // OLD (rollback reference): diff_mu = 0;
            }
            else
            {
                Value = pGpmu.segment<3>(v * 3);
            }

            // OLD (rollback reference): write(indices[v], diff_mu, Value);
            pGpP(offset++).write(indices[v],  // Vid I
                                 diff_mu < 0 ? 0 : diff_mu,  // Safe zero column
                                 Value);
        }
    }
}  // namespace detail


class IPCSimplexFrictionalContactDiffParm final : public SimplexFrictionalContactDiffParm
{
  public:
    using SimplexFrictionalContactDiffParm::SimplexFrictionalContactDiffParm;

    // Inherited via SimplexFrictionalContactDiffParm
    void do_build(BuildInfo& info) override {}

    void do_init(InitInfo&) override {}

    void do_assemble(DiffParmInfo& info) override
    {
        using namespace muda;
        namespace CIC = sym::codim_ipc_contact;
        namespace DF  = sym::codim_ipc_diff_friction;

        auto PT_count = info.friction_PTs().size();
        if(PT_count)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    PT_count,
                    [pGpPs = info.PT_pGpP().viewer().name("pGpPs"),
                     table = info.contact_tabular().viewer().name("table"),
                     diff_table = info.diff_contact_tabular().viewer().name("diff_table"),
                     contact_ids = info.contact_element_ids().viewer().name("contact_element_ids"),
                     PTs     = info.friction_PTs().viewer().name("PTs"),
                     Ps      = info.positions().viewer().name("Ps"),
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
                        Vector12 pGpmu;


                        //DF::PT_friction_pGpkappa_pGpmu(
                        //    pGpkappa, pGpmu, kappa, mu, X, Xn, d_hat, thickness, eps_vh);

                        namespace CIC = sym::codim_ipc_contact;
                        Eigen::Matrix<Float, 2, 12> T;
                        sym::codim_ipc_diff_friction::PT::TR(T, Xn);

                        Vector12 dX  = X - Xn;
                        Vector2  tau = T * dX;

                        Float D;
                        distance::point_triangle_distance2<Float>(Xn.segment<3>(0),
                                                                  Xn.segment<3>(3),
                                                                  Xn.segment<3>(6),
                                                                  Xn.segment<3>(9),
                                                                  D);

                        Vector2 pGpkappa2, pGpmu2;
                        CIC::d_friction_gradient_d_kappa_mu(
                            pGpkappa2, pGpmu2, kappa, mu, tau, eps_vh, d_hat, thickness, D);


                        pGpkappa = pGpkappa2.transpose() * T;
                        pGpmu    = pGpmu2.transpose() * T;

                        pGpkappa *= dt2;
                        pGpmu *= dt2;


                        detail::assemble_pGpP<4>(I, pGpPs, pGpkappa, pGpmu, diff_coeff, PT);
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
                        Vector12 pGpmu;

                        if(distance::need_mollify<Float>(Xn.segment<3>(0),
                                                         Xn.segment<3>(3),
                                                         Xn.segment<3>(6),
                                                         Xn.segment<3>(9),
                                                         eps_x))
                        {
                            pGpkappa.setZero();
                            pGpmu.setZero();
                        }
                        else
                        {
                            Float eps_vh = eps_v * dt;

                            DF::EE_friction_pGpkappa_pGpmu(
                                pGpkappa, pGpmu, kappa, mu, X, Xn, d_hat, thickness, eps_vh);

                            auto dt2 = dt * dt;

                            pGpkappa *= dt2;
                            pGpmu *= dt2;
                        }

                        detail::assemble_pGpP<4>(I, pGpPs, pGpkappa, pGpmu, diff_coeff, EE);
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

                           Float dt2    = dt * dt;
                           Float eps_vh = eps_v * dt;

                           Vector9 pGpkappa;
                           Vector9 pGpmu;

                           DF::PE_friction_pGpkappa_pGpmu(
                               pGpkappa, pGpmu, kappa, mu, X, Xn, d_hat, thickness, eps_vh);

                           pGpkappa *= dt2;
                           pGpmu *= dt2;

                           detail::assemble_pGpP<3>(I, pGpPs, pGpkappa, pGpmu, diff_coeff, PE);
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
                    Vector6 pGpmu;

                    DF::PP_friction_pGpkappa_pGpmu(
                        pGpkappa, pGpmu, kappa, mu, X, Xn, d_hat, thickness, eps_vh);

                    pGpkappa *= dt2;
                    pGpmu *= dt2;

                    detail::assemble_pGpP<2>(I, pGpPs, pGpkappa, pGpmu, diff_coeff, PP);
                });
        }
    }
};

REGISTER_SIM_SYSTEM(IPCSimplexFrictionalContactDiffParm);
}  // namespace uipc::backend::cuda