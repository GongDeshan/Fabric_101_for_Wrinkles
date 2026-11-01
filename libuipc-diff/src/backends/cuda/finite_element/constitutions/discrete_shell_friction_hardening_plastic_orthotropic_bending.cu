#include <finite_element/finite_element_extra_constitution.h>
#include <uipc/builtin/attribute_name.h>
#include <finite_element/constitutions/discrete_shell_friction_hardening_plastic_orthotropic_bending_function.h>
#include <numbers>
#include <utils/matrix_assembly_utils.h>
#include <kernel_cout.h>
#include <finite_element/finite_element_extra_constitution_diff_parm_reporter.h>
#include <finite_element/finite_element_diff_dof_reporter.h>
#include <sim_engine.h>
#include <linear_system/global_linear_system.h>

namespace std
{
// hash function for Vector2i
template <>
struct hash<uipc::Vector2i>
{
    size_t operator()(const uipc::Vector2i& v) const
    {
        size_t front = v[0];
        size_t end   = v[1];
        return front << 32 | end;
    }
};
}  // namespace std

namespace uipc::backend::cuda
{
static constexpr U64 DiscreteShellBendingUID = 24;

// Rest bias is independent of positions and optimized material parameters.
// Scale energy/residual derivatives once, leaving physical V_bar unchanged.
// Equal warp/weft materials then match isotropic bending for compatible
// initial histories and zero torsion/Poisson terms.
inline UIPC_GENERIC Float orthotropic_bending_normalization(Float bias_angle)
{
    Float c = cos(bias_angle + 1.570796);
    Float sn = sin(bias_angle + 1.570796);
    Float w = c * c, v = sn * sn;
    return 1.0 / (w * w + v * v);
}


class DiscreteShellFrictionHardeningPlasticOrthotropicBending final : public FiniteElementExtraConstitution
{
    using Base = FiniteElementExtraConstitution;
    friend class DiscreteShellFrictionHardeningPlasticOrthotropicBending;

  public:
    using Base::Base;

    virtual U64 get_uid() const noexcept override
    {
        return DiscreteShellBendingUID;
    }

    class InitInfo
    {
      public:
        bool        valid_bending() const { return oppo_verts.size() == 2; }
        IndexT      edge_index    = -1;
        IndexT      stencil_index = -1;
        set<IndexT> oppo_verts;
    };

    vector<Vector4i> h_stencils;  // X0, X1, X2, X3; (X1, X2) is middle edge
    vector<Float>    h_bending_stiffness_warp;
    vector<Float>    h_bending_stiffness_weft;
    vector<Float>    h_bending_torsion;
    vector<Float>    h_bending_poisson;
    vector<Float>    h_rest_volumes;
    vector<Float>    h_rest_lengths;
    vector<Float>    h_theta_bars;
    vector<Float>    h_theta_torsion_bar;
    vector<Float> h_bias_angles;  // The Angle of edge between warp in uv coordinate
    vector<Float> h_h_bars;
    vector<Float> h_V_bars;

    // For (perfect) plastic parameters
    vector<Float> h_hardening_ks_warp;
    vector<Float> h_hardening_ks_weft;
    vector<Float> h_hardening_plastic_thetas_warp;
    vector<Float> h_hardening_plastic_thetas_weft;
    vector<Float> h_dihedral_thetas;
    vector<Float> h_plastic_thetas_warp;
    vector<Float> h_plastic_thetas_weft;
    vector<Float> h_yield_thetas_warp;
    vector<Float> h_yield_thetas_weft;

    // For (Columb) friction parameters
    vector<Float> h_friction_stiffness_warp;
    vector<Float> h_friction_stiffness_weft;
    vector<Float> h_thres_thetas_warp;
    vector<Float> h_thres_thetas_weft;
    vector<Float> h_anchor_thetas_warp;
    vector<Float> h_anchor_thetas_weft;

    muda::DeviceBuffer<Vector4i> stencils;  // X0, X1, X2, X3; (X1, X2) is middle edge
    muda::DeviceBuffer<Float>         bending_stiffnesses_warp;
    muda::DeviceBuffer<Float>         bending_stiffnesses_weft;
    muda::DeviceBuffer<Float>         bending_torsions;
    muda::DeviceBuffer<Float>         bending_poissons;
    muda::DeviceBuffer<Float>         rest_lengths;
    muda::DeviceBuffer<Float>         h_bars;
    muda::DeviceBuffer<Float>         theta_bars;
    muda::DeviceBuffer<Float>         theta_torsion_bars;
    muda::DeviceBuffer<Float>         V_bars;
    muda::DeviceBuffer<Float>         bias_angles;
    unordered_map<Vector2i, InitInfo> stencil_map;  // Edge -> opposite vertices









    muda::DeviceBuffer<Float> hardening_ks_warp;
    muda::DeviceBuffer<Float> hardening_ks_weft;
    muda::DeviceBuffer<Float> hardening_plastic_thetes_warp;
    muda::DeviceBuffer<Float> hardening_plastic_thetes_weft;



    muda::DeviceBuffer<Float> d_hardening_plastic_theta_d_yield_ori_warp;
    muda::DeviceBuffer<Float> d_hardening_plastic_theta_d_yield_ori_weft;
    muda::DeviceBuffer<Float> d_hardening_plastic_theta_d_k_warp;
    muda::DeviceBuffer<Float> d_hardening_plastic_theta_d_k_weft;
    muda::DeviceBuffer<Float> d_hardening_plastic_theta_d_hardening_k_warp;
    muda::DeviceBuffer<Float> d_hardening_plastic_theta_d_hardening_k_weft;

    muda::DeviceBuffer<Float> forward_d_plastic_theta_d_yield_ori_warp;
    muda::DeviceBuffer<Float> forward_d_plastic_theta_d_k_warp;
    muda::DeviceBuffer<Float> forward_d_plastic_theta_d_hardening_k_warp;
    muda::DeviceBuffer<Float> forward_d_anchor_theta_d_thres_theta_warp;
    muda::DeviceBuffer<Float> forward_d_plastic_theta_d_yield_ori_weft;
    muda::DeviceBuffer<Float> forward_d_plastic_theta_d_k_weft;
    muda::DeviceBuffer<Float> forward_d_plastic_theta_d_hardening_k_weft;
    muda::DeviceBuffer<Float> forward_d_anchor_theta_d_thres_theta_weft;
    muda::DeviceBuffer<Float> dihedral_thetas;
    muda::DeviceBuffer<Float> plastic_thetas_warp;
    muda::DeviceBuffer<Float> plastic_thetas_weft;
    muda::DeviceBuffer<Float> yield_ori_thetas_warp, yield_thetas_warp;
    muda::DeviceBuffer<Float> yield_ori_thetas_weft, yield_thetas_weft;

    // Friction Parameter in GPU
    muda::DeviceBuffer<Float> friction_stiffnesses_warp;
    muda::DeviceBuffer<Float> friction_stiffnesses_weft;
    muda::DeviceBuffer<Float> thres_thetas_warp;
    muda::DeviceBuffer<Float> thres_thetas_weft;
    muda::DeviceBuffer<Float> anchor_thetas_warp;
    muda::DeviceBuffer<Float> anchor_thetas_weft;

    FiniteElementMethod*                 fem;
    GlobalLinearSystem*                  audit_linear_system = nullptr;
    std::span<S<geometry::GeometrySlot>> geo_slots;
    vector<bool>                         is_valid_bending;

    vector<IndexT> filter_geo_slot_indices;

    // Forward-only snapshots: replay must never overwrite the state transition
    // at t (which consumes X[t-1]). Initial X[0] is not an optimized dof.
    struct HistoryFrame
    {
        vector<Vector12> plastic_transition;  // two [A00,A01,A10,A11,bp,bq]
        vector<Vector4>  anchor_transition;   // two [inherit,b_theta]
        vector<Vector12> source_gradient;
        vector<Vector4>  force_coefficients;  // dp_w, dp_f, da_w, da_f; no dt^2
    };
    vector<HistoryFrame>         position_history;
    GlobalDiffSimManager*        history_manager = nullptr;
    bool history_position_gradient_enabled = true;
    muda::DeviceBuffer<Vector12> history_plastic, history_gradient;
    muda::DeviceBuffer<Vector4>  history_anchor, history_coefficients;

    void record_position_history()
    {
        const SizeT frame = engine().frame();
        if(frame == 1)
            position_history.clear();
        if(!history_position_gradient_enabled || !history_manager
           || !history_manager->is_grad_enabled())
            return;
        const SizeT N = stencils.size();
        history_plastic.resize(N);
        history_gradient.resize(N);
        history_anchor.resize(N);
        history_coefficients.resize(N);
        using namespace muda;
        using namespace sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(N,
                   [idxs = stencils.cviewer().name("history_stencils"),
                    xs   = fem->xs().cviewer().name("history_xs"),
                    bias = bias_angles.cviewer().name("history_bias"),
                    kw = bending_stiffnesses_warp.cviewer().name("history_kw"),
                    kf = bending_stiffnesses_weft.cviewer().name("history_kf"),
                    kp = bending_poissons.cviewer().name("history_kp"),
                    fw = friction_stiffnesses_warp.cviewer().name("history_fw"),
                    ff = friction_stiffnesses_weft.cviewer().name("history_ff"),
                    hw = hardening_ks_warp.cviewer().name("history_hw"),
                    hf = hardening_ks_weft.cviewer().name("history_hf"),
                    pw = plastic_thetas_warp.cviewer().name("history_pw"),
                    pf = plastic_thetas_weft.cviewer().name("history_pf"),
                    qw = hardening_plastic_thetes_warp.cviewer().name("history_qw"),
                    qf = hardening_plastic_thetes_weft.cviewer().name("history_qf"),
                    yw = yield_ori_thetas_warp.cviewer().name("history_yw"),
                    yf = yield_ori_thetas_weft.cviewer().name("history_yf"),
                    aw = anchor_thetas_warp.cviewer().name("history_aw"),
                    af = anchor_thetas_weft.cviewer().name("history_af"),
                    tw = thres_thetas_warp.cviewer().name("history_tw"),
                    tf = thres_thetas_weft.cviewer().name("history_tf"),
                    L  = rest_lengths.cviewer().name("history_L"),
                    hb = h_bars.cviewer().name("history_hbar"),
                    V  = V_bars.cviewer().name("history_V"),
                    A  = history_plastic.viewer().name("history_A"),
                    B  = history_anchor.viewer().name("history_B"),
                    G  = history_gradient.viewer().name("history_G"),
                    C = history_coefficients.viewer().name("history_C")] __device__(int I)
                   {
                       auto  idx = idxs(I);
                       Float theta;
                       dihedral_angle(xs(idx[0]), xs(idx[1]), xs(idx[2]), xs(idx[3]), theta);
                       Vector12 g;
                       dihedral_angle_gradient(
                           xs(idx[0]), xs(idx[1]), xs(idx[2]), xs(idx[3]), g);
                       G(I)                = g;
                       Float    c          = cos(bias(I) + 1.570796);
                       Float    s          = sin(bias(I) + 1.570796);
                       Float    weights[2] = {c * c, s * s};
                       Float    ks[2] = {kw(I), kf(I)}, hs[2] = {hw(I), hf(I)};
                       Float    ps[2] = {pw(I), pf(I)}, qs[2] = {qw(I), qf(I)};
                       Float    ys[2] = {yw(I), yf(I)}, as[2] = {aw(I), af(I)};
                       Float y0 = weights[0] * yw(I) + weights[1] * yf(I);
                       Float t0 = weights[0] * tw(I) + weights[1] * tf(I);
                       ys[0] = weights[0] * y0;
                       ys[1] = weights[1] * y0;
                       Float ts[2] = {weights[0] * t0, weights[1] * t0};
                       Vector12 transition = Vector12::Zero();
                       Vector4  anchor     = Vector4::Zero();
                       for(int d = 0; d < 2; ++d)
                       {
                           int o         = 6 * d;
                           transition[o] = transition[o + 3] = 1.0;
                           anchor[2 * d]                     = 1.0;
                           Float e = theta * weights[d] - ps[d];
                           if(abs(e) > ys[d] + qs[d] * hs[d] / ks[d])
                           {
                               Float r         = ks[d] / (ks[d] + hs[d]);
                               Float direction = sign(e);
                               transition[o]   = 1.0 - r;
                               transition[o + 1] = -direction * r * hs[d] / ks[d];
                               transition[o + 2] = -direction * r;
                               transition[o + 3] = 1.0 - r * hs[d] / ks[d];
                               transition[o + 4] = r * weights[d];
                               transition[o + 5] = direction * r * weights[d];
                           }
                           if(abs(theta * weights[d] - as[d]) > ts[d])
                           {
                               anchor[2 * d]     = 0.0;
                               anchor[2 * d + 1] = weights[d];
                           }
                       }
                       A(I)        = transition;
                       B(I)        = anchor;
                       Float scale = -2.0 * L(I) / hb(I) * V(I)
                           * orthotropic_bending_normalization(bias(I));
                       C(I) << scale * (kw(I) * weights[0] + kp(I) * weights[1]),
                           scale * (kf(I) * weights[1] + kp(I) * weights[0]),
                           scale * fw(I) * weights[0], scale * ff(I) * weights[1];
                   });
        // Small synchronous transfers retain O(T*N) transition storage without
        // coupling forward state to the replayer. Assemble only nonzero chains.
        UIPC_ASSERT(frame > 0 && position_history.size() == frame - 1,
                    "Bending history requires a complete grad-enabled rollout from frame 1");
        HistoryFrame snapshot;
        snapshot.plastic_transition.resize(N);
        snapshot.anchor_transition.resize(N);
        snapshot.source_gradient.resize(N);
        snapshot.force_coefficients.resize(N);
        history_plastic.view().copy_to(snapshot.plastic_transition.data());
        history_anchor.view().copy_to(snapshot.anchor_transition.data());
        history_gradient.view().copy_to(snapshot.source_gradient.data());
        history_coefficients.view().copy_to(snapshot.force_coefficients.data());
        position_history.push_back(std::move(snapshot));
    }

    void friction_plastic_update()
    {
        //std::cout << "friction plastic update start" << std::endl;

        using namespace muda;
        using namespace sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        // Used when single object
        auto geo = geo_slots.front()->geometry().as<geometry::SimplicialComplex>();

        // Used when sim multiple objects
        //auto geo = geo_slots[filter_geo_slot_indices[0]]->geometry().as<geometry::SimplicialComplex>();

        auto rest_anchor_plastic =
            geo->edges().find<I32>("reset_anchor_plastic")->view();

        if(rest_anchor_plastic[0] != 0)
        {
            auto plastic_thetas_warp_view =
                geo->edges().find<Float>("plastic_theta_warp")->view();
            auto anchor_thetas_warp_view =
                geo->edges().find<Float>("anchor_theta_warp")->view();
            //auto yield_theta_warp_view =
            //    geo->edges().find<Float>("yield_theta_warp")->view();

            auto plastic_thetas_weft_view =
                geo->edges().find<Float>("plastic_theta_weft")->view();
            auto anchor_thetas_weft_view =
                geo->edges().find<Float>("anchor_theta_weft")->view();
            //auto yield_theta_weft_view =
            //    geo->edges().find<Float>("yield_theta_weft")->view();

            int idx_cnt{0};

            int cnt{0};

            for(int i = 0; i < is_valid_bending.size(); ++i)
            {
                if(is_valid_bending[i])
                {
                    h_anchor_thetas_warp[idx_cnt] = anchor_thetas_warp_view[cnt];
                    h_anchor_thetas_weft[idx_cnt] = anchor_thetas_weft_view[cnt];

                    //h_yield_thetas_warp[idx_cnt] = yield_theta_warp_view[cnt];
                    //h_yield_thetas_weft[idx_cnt] = yield_theta_weft_view[cnt];

                    h_plastic_thetas_warp[idx_cnt] = plastic_thetas_warp_view[cnt];
                    h_plastic_thetas_weft[idx_cnt] = plastic_thetas_weft_view[cnt];

                    idx_cnt++;
                }
                cnt++;
            }

            anchor_thetas_warp.view().copy_from(h_anchor_thetas_warp.data());
            anchor_thetas_weft.view().copy_from(h_anchor_thetas_weft.data());

            //yield_thetas_warp.view().copy_from(h_yield_thetas_warp.data());
            //yield_thetas_weft.view().copy_from(h_yield_thetas_weft.data());

            yield_thetas_warp = yield_ori_thetas_warp;
            yield_thetas_weft = yield_ori_thetas_weft;

            plastic_thetas_warp.view().copy_from(h_plastic_thetas_warp.data());
            plastic_thetas_weft.view().copy_from(h_plastic_thetas_weft.data());

            hardening_plastic_thetes_warp.fill(0.0);
            hardening_plastic_thetes_weft.fill(0.0);

            // A reset starts a new trajectory.  State sensitivities must be
            // reset together with the return-mapping state itself.
            forward_d_plastic_theta_d_yield_ori_warp.fill(0.0);
            forward_d_plastic_theta_d_k_warp.fill(0.0);
            forward_d_plastic_theta_d_hardening_k_warp.fill(0.0);
            forward_d_anchor_theta_d_thres_theta_warp.fill(0.0);
            forward_d_plastic_theta_d_yield_ori_weft.fill(0.0);
            forward_d_plastic_theta_d_k_weft.fill(0.0);
            forward_d_plastic_theta_d_hardening_k_weft.fill(0.0);
            forward_d_anchor_theta_d_thres_theta_weft.fill(0.0);
            d_hardening_plastic_theta_d_yield_ori_warp.fill(0.0);
            d_hardening_plastic_theta_d_yield_ori_weft.fill(0.0);
            d_hardening_plastic_theta_d_k_warp.fill(0.0);
            d_hardening_plastic_theta_d_k_weft.fill(0.0);
            d_hardening_plastic_theta_d_hardening_k_warp.fill(0.0);
            d_hardening_plastic_theta_d_hardening_k_weft.fill(0.0);
            fem->d_plastic_theta_d_yield_ori_warp.fill(0.0);
            fem->d_plastic_theta_d_yield_ori_weft.fill(0.0);
            fem->d_plastic_theta_d_k_warp.fill(0.0);
            fem->d_plastic_theta_d_k_weft.fill(0.0);
            fem->d_plastic_theta_d_hardening_k_warp.fill(0.0);
            fem->d_plastic_theta_d_hardening_k_weft.fill(0.0);
            fem->d_anchor_theta_d_thres_theta_warp.fill(0.0);
            fem->d_anchor_theta_d_thres_theta_weft.fill(0.0);
        }

        record_position_history();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                stencils.size(),
                [stencils    = stencils.viewer().name("stencils"),
                 bias_angles = bias_angles.cviewer().name("bias_angles"),
                 bending_stiffnesses_warp = bending_stiffnesses_warp.viewer().name("bending_stiffness_warp"),
                 bending_stiffnesses_weft = bending_stiffnesses_weft.viewer().name("bending_stiffness_weft"),
                 dihedral_thetas = dihedral_thetas.viewer().name("dihedral_theta"),
                 anchor_thetas_warp = anchor_thetas_warp.viewer().name("anchor_theta_warp"),
                 anchor_thetas_weft = anchor_thetas_weft.viewer().name("anchor_theta_weft"),
                 hardening_ks_warp = hardening_ks_warp.viewer().name("hardening_k_warp"),
                 hardening_ks_weft = hardening_ks_weft.viewer().name("hardening_k_weft"),
                 yield_thetas_warp = yield_thetas_warp.viewer().name("yield_theta_warp"),
                 yield_thetas_weft = yield_thetas_weft.viewer().name("yield_theta_weft"),
                 yield_ori_thetas_warp = yield_ori_thetas_warp.viewer().name("yield_ori_theta_warp"),
                 yield_ori_thetas_weft = yield_ori_thetas_weft.viewer().name("yield_ori_theta_weft"),
                 hardening_plastic_thetas_warp =
                     hardening_plastic_thetes_warp.viewer().name("hardening_plastic_theta_warp"),
                 hardening_plastic_thetas_weft =
                     hardening_plastic_thetes_weft.viewer().name("hardening_plastic_theta_weft"),
                 d_hardening_plastic_theta_d_yield_ori_warp =
                     d_hardening_plastic_theta_d_yield_ori_warp.viewer().name(
                         "d_hardening_plastic_theta_d_yield_ori_warp"),
                 d_hardening_plastic_theta_d_yield_ori_weft =
                     d_hardening_plastic_theta_d_yield_ori_weft.viewer().name(
                         "d_hardening_plastic_theta_d_yield_ori_weft"),
                 d_hardening_plastic_theta_d_k_warp =
                     d_hardening_plastic_theta_d_k_warp.viewer().name("d_hardening_plastic_theta_d_k_warp"),
                 d_hardening_plastic_theta_d_k_weft =
                     d_hardening_plastic_theta_d_k_weft.viewer().name("d_hardening_plastic_theta_d_k_weft"),
                 d_hardening_plastic_theta_d_hardening_k_warp =
                     d_hardening_plastic_theta_d_hardening_k_warp.viewer().name(
                         "d_hardening_plastic_theta_d_hardening_k_warp"),
                 d_hardening_plastic_theta_d_hardening_k_weft =
                     d_hardening_plastic_theta_d_hardening_k_weft.viewer().name(
                         "d_hardening_plastic_theta_d_hardening_k_weft"),
                 plastic_thetas_warp = plastic_thetas_warp.viewer().name("plastic_theta_warp"),
                 plastic_thetas_weft = plastic_thetas_weft.viewer().name("plastic_theta_weft"),
                 thres_thetas_warp = thres_thetas_warp.viewer().name("thres_theta_warp"),
                 thres_thetas_weft = thres_thetas_weft.viewer().name("thres_theta_weft"),
                 fem_plastic_thetas_warp = fem->plastic_thetas_warp.viewer().name("fem_plastic_theta_warp"),
                 recorded_d_plastic_theta_d_yield_ori_warp =
                     fem->d_plastic_theta_d_yield_ori_warp.viewer().name("recorded_d_plastic_theta_d_yield_ori_warp"),
                 recorded_d_plastic_theta_d_k_warp =
                     fem->d_plastic_theta_d_k_warp.viewer().name("recorded_d_plastic_theta_d_k_warp"),
                 recorded_d_plastic_theta_d_hardening_k_warp =
                     fem->d_plastic_theta_d_hardening_k_warp.viewer().name(
                         "recorded_d_plastic_theta_d_hardening_k_warp"),
                 recorded_d_anchor_theta_d_thres_theta_warp =
                     fem->d_anchor_theta_d_thres_theta_warp.viewer().name(
                         "recorded_d_anchor_theta_d_thres_theta_warp"),
                 recorded_d_plastic_theta_d_yield_ori_weft =
                     fem->d_plastic_theta_d_yield_ori_weft.viewer().name("recorded_d_plastic_theta_d_yield_ori_weft"),
                 recorded_d_plastic_theta_d_k_weft =
                     fem->d_plastic_theta_d_k_weft.viewer().name("recorded_d_plastic_theta_d_k_weft"),
                 recorded_d_plastic_theta_d_hardening_k_weft =
                     fem->d_plastic_theta_d_hardening_k_weft.viewer().name(
                         "recorded_d_plastic_theta_d_hardening_k_weft"),
                 recorded_d_anchor_theta_d_thres_theta_weft =
                     fem->d_anchor_theta_d_thres_theta_weft.viewer().name(
                         "recorded_d_anchor_theta_d_thres_theta_weft"),
                 fem_plastic_thetas_weft = fem->plastic_thetas_weft.viewer().name("fem_plastic_theta_weft"),
                 fem_anchor_thetas_warp = fem->anchor_thetas_warp.viewer().name("fem_anchor_theta_warp"),
                 fem_anchor_thetas_weft = fem->anchor_thetas_weft.viewer().name("fem_anchor_theta_weft"),
                 fem_d_anchor_theta_d_thres_theta_warp =
                     forward_d_anchor_theta_d_thres_theta_warp.viewer().name("fem_d_anchor_theta_d_thres_theta_warp"),
                 fem_d_anchor_theta_d_thres_theta_weft =
                     forward_d_anchor_theta_d_thres_theta_weft.viewer().name("fem_d_anchor_theta_d_thres_theta_weft"),
                 fem_d_plastic_theta_d_yield_ori_warp =
                     forward_d_plastic_theta_d_yield_ori_warp.viewer().name("fem_d_plastic_theta_d_yield_ori_warp"),
                 fem_d_plastic_theta_d_yield_ori_weft =
                     forward_d_plastic_theta_d_yield_ori_weft.viewer().name("fem_d_plastic_theta_d_yield_ori_weft"),
                 fem_d_plastic_theta_d_k_warp =
                     forward_d_plastic_theta_d_k_warp.viewer().name("fem_d_plastic_theta_d_k_warp"),
                 fem_d_plastic_theta_d_k_weft =
                     forward_d_plastic_theta_d_k_weft.viewer().name("fem_d_plastic_theta_d_k_weft"),
                 fem_d_plastic_theta_d_hardening_k_warp =
                     forward_d_plastic_theta_d_hardening_k_warp.viewer().name(
                         "fem_d_plastic_theta_d_hardening_k_warp"),
                 fem_d_plastic_theta_d_hardening_k_weft =
                     forward_d_plastic_theta_d_hardening_k_weft.viewer().name(
                         "fem_d_plastic_theta_d_hardening_k_weft"),
                 xs = fem->xs().viewer().name("xs")] __device__(int I)
                {
                    Vector4i stencil = stencils(I);

                    Float bias_angle = bias_angles(I);

                    Float k_warp = bending_stiffnesses_warp(I);
                    Float k_weft = bending_stiffnesses_weft(I);

                    Float hk_warp = hardening_ks_warp(I);
                    Float hk_weft = hardening_ks_weft(I);

                    Float anchor_theta_warp = anchor_thetas_warp(I);
                    Float anchor_theta_weft = anchor_thetas_weft(I);

                    Float thres_theta_warp = thres_thetas_warp(I);
                    Float thres_theta_weft = thres_thetas_weft(I);

                    Float plastic_theta_warp = plastic_thetas_warp(I);
                    Float plastic_theta_weft = plastic_thetas_weft(I);

                    Float hardening_plastic_theta_warp =
                        hardening_plastic_thetas_warp(I);
                    Float hardening_plastic_theta_weft =
                        hardening_plastic_thetas_weft(I);

                    //Float yield_theta_warp = yield_thetas_warp(I);
                    //Float yield_theta_weft = yield_thetas_weft(I);

                    Float yield_ori_theta_warp = yield_ori_thetas_warp(I);
                    Float yield_ori_theta_weft = yield_ori_thetas_weft(I);

                    //Float yield_theta_warp =
                    //    yield_ori_theta_warp
                    //    + hardening_plastic_theta_warp * (hk_warp / k_warp);
                    //Float yield_theta_weft =
                    //    yield_ori_theta_weft
                    //    + hardening_plastic_theta_weft * (hk_weft / k_weft);

                    Vector3 x0 = xs(stencil[0]);
                    Vector3 x1 = xs(stencil[1]);
                    Vector3 x2 = xs(stencil[2]);
                    Vector3 x3 = xs(stencil[3]);
                    Float   theta;
                    dihedral_angle(x0, x1, x2, x3, theta);

                    dihedral_thetas(I) = theta;

                    Float angle_between_warp = bias_angle + 1.570796;
                    Float cos_angle          = cos(angle_between_warp);
                    Float sin_angle          = sin(angle_between_warp);

                    Float theta_warp    = theta * cos_angle * cos_angle;
                    Float theta_weft    = theta * sin_angle * sin_angle;
                    Float theta_torsion = theta * cos_angle * sin_angle;

                    Float elastic_theta_warp = theta_warp - plastic_theta_warp;
                    Float elastic_theta_weft = theta_weft - plastic_theta_weft;

                    Float thres_theta_eff = thres_theta_warp * cos_angle * cos_angle
                                            + thres_theta_weft * sin_angle * sin_angle;

                    Float thres_theta_warp_eff = thres_theta_eff * cos_angle * cos_angle;
                    Float thres_theta_weft_eff = thres_theta_eff * sin_angle * sin_angle;

                    Float yield_ori_theta_eff =
                        yield_ori_theta_warp * cos_angle * cos_angle
                        + yield_ori_theta_weft * sin_angle * sin_angle;

                    Float yield_ori_theta_warp_eff =
                        yield_ori_theta_eff * cos_angle * cos_angle;
                    Float yield_ori_theta_weft_eff =
                        yield_ori_theta_eff * sin_angle * sin_angle;

                    Float yield_theta_warp_eff =
                        yield_ori_theta_warp_eff
                        + hardening_plastic_theta_warp * (hk_warp / k_warp);
                    Float yield_theta_weft_eff =
                        yield_ori_theta_weft_eff
                        + hardening_plastic_theta_weft * (hk_weft / k_weft);

                    //bool is_yield_warp = abs(elastic_theta_warp) > yield_theta_warp;
                    //bool is_yield_weft = abs(elastic_theta_weft) > yield_theta_weft;

                    bool is_yield_warp = abs(elastic_theta_warp) > yield_theta_warp_eff;
                    bool is_yield_weft = abs(elastic_theta_weft) > yield_theta_weft_eff;

                    if(is_yield_warp)
                    {
                        //plastic_thetas_warp(I) =
                        //    plastic_theta_warp
                        //    + sign(elastic_theta_warp)
                        //          * (abs(elastic_theta_warp) - yield_theta_warp)
                        //          * (k_warp / (k_warp + hk_warp));

                        //hardening_plastic_thetas_warp(I) =
                        //    hardening_plastic_theta_warp
                        //    + (abs(elastic_theta_warp) - yield_theta_warp)
                        //          * (k_warp / (k_warp + hk_warp));

                        plastic_thetas_warp(I) =
                            plastic_theta_warp
                            + sign(elastic_theta_warp)
                                  * (abs(elastic_theta_warp) - yield_theta_warp_eff)
                                  * (k_warp / (k_warp + hk_warp));

                        hardening_plastic_thetas_warp(I) =
                            hardening_plastic_theta_warp
                            + (abs(elastic_theta_warp) - yield_theta_warp_eff)
                                  * (k_warp / (k_warp + hk_warp));








                        Float s_warp = sign(elastic_theta_warp);
                        Float r_warp = k_warp / (k_warp + hk_warp);
                        Float f_warp = abs(elastic_theta_warp) - yield_theta_warp_eff;
                        Float denom_sq_warp = (k_warp + hk_warp) * (k_warp + hk_warp);

                        Float dp_dy_warp = fem_d_plastic_theta_d_yield_ori_warp(I);
                        Float dp_dk_warp = fem_d_plastic_theta_d_k_warp(I);
                        Float dp_dh_warp = fem_d_plastic_theta_d_hardening_k_warp(I);
                        Float dq_dy_warp = d_hardening_plastic_theta_d_yield_ori_warp(I);
                        Float dq_dk_warp = d_hardening_plastic_theta_d_k_warp(I);
                        Float dq_dh_warp =
                            d_hardening_plastic_theta_d_hardening_k_warp(I);

                        Float dgamma_dy_warp =
                            r_warp * (-s_warp * dp_dy_warp - (1.0 + (hk_warp / k_warp) * dq_dy_warp));
                        Float dgamma_dk_warp =
                            r_warp
                                * (-s_warp * dp_dk_warp
                                   - ((hk_warp / k_warp) * dq_dk_warp
                                      - hardening_plastic_theta_warp * hk_warp
                                            / (k_warp * k_warp)))
                            + f_warp * hk_warp / denom_sq_warp;
                        Float dgamma_dh_warp =
                            r_warp
                                * (-s_warp * dp_dh_warp
                                   - ((hk_warp / k_warp) * dq_dh_warp
                                      + hardening_plastic_theta_warp / k_warp))
                            - f_warp * k_warp / denom_sq_warp;

                        fem_d_plastic_theta_d_yield_ori_warp(I) =
                            dp_dy_warp + s_warp * dgamma_dy_warp;
                        fem_d_plastic_theta_d_k_warp(I) = dp_dk_warp + s_warp * dgamma_dk_warp;
                        fem_d_plastic_theta_d_hardening_k_warp(I) =
                            dp_dh_warp + s_warp * dgamma_dh_warp;
                        d_hardening_plastic_theta_d_yield_ori_warp(I) =
                            dq_dy_warp + dgamma_dy_warp;
                        d_hardening_plastic_theta_d_k_warp(I) = dq_dk_warp + dgamma_dk_warp;
                        d_hardening_plastic_theta_d_hardening_k_warp(I) =
                            dq_dh_warp + dgamma_dh_warp;
                    }
                    // In the elastic branch p_new=p and q_new=q, therefore all
                    // state sensitivities are inherited unchanged.

                    //printf("Forward fem_d_plastic_theta_d_yield_ori_warp: %f\r\n",
                    //       fem_d_plastic_theta_d_yield_ori_warp(I));

                    //printf("is_yield_weft: %f\r\n", is_yield_weft);
                    //printf("elastic_theta_weft: %f\n", elastic_theta_weft);
                    //printf("plastic_theta_weft: %f\n", plastic_theta_weft);
                    //printf("yield_theta_weft: %f\n", yield_theta_weft);
                    //printf("yield_ori_theta_weft: %f\n", yield_ori_theta_weft);
                    //printf("k_weft: %f\n", k_weft);
                    //printf("hk_weft: %f\n", hk_weft);
                    //printf("hardening_plastic_theta_weft: %f\r\n", hardening_plastic_theta_weft);

                    //if(is_yield_weft)
                    //{
                    //    printf("weft yeild\r\n");
                    //}
                    //else
                    //{
                    //    printf("weft not yeild\r\n");
                    //
                    //}

                    if(is_yield_weft)
                    {
                        //plastic_thetas_weft(I) =
                        //    plastic_theta_weft
                        //    + sign(elastic_theta_weft)
                        //          * (abs(elastic_theta_weft) - yield_theta_weft)
                        //          * (k_weft / (k_weft + hk_weft));

                        //hardening_plastic_thetas_weft(I) =
                        //    hardening_plastic_theta_weft
                        //    + (abs(elastic_theta_weft) - yield_theta_weft)
                        //          * (k_weft / (k_weft + hk_weft));

                        plastic_thetas_weft(I) =
                            plastic_theta_weft
                            + sign(elastic_theta_weft)
                                  * (abs(elastic_theta_weft) - yield_theta_weft_eff)
                                  * (k_weft / (k_weft + hk_weft));

                        hardening_plastic_thetas_weft(I) =
                            hardening_plastic_theta_weft
                            + (abs(elastic_theta_weft) - yield_theta_weft_eff)
                                  * (k_weft / (k_weft + hk_weft));



                        Float s_weft = sign(elastic_theta_weft);
                        Float r_weft = k_weft / (k_weft + hk_weft);
                        Float f_weft = abs(elastic_theta_weft) - yield_theta_weft_eff;
                        Float denom_sq_weft = (k_weft + hk_weft) * (k_weft + hk_weft);

                        Float dp_dy_weft = fem_d_plastic_theta_d_yield_ori_weft(I);
                        Float dp_dk_weft = fem_d_plastic_theta_d_k_weft(I);
                        Float dp_dh_weft = fem_d_plastic_theta_d_hardening_k_weft(I);
                        Float dq_dy_weft = d_hardening_plastic_theta_d_yield_ori_weft(I);
                        Float dq_dk_weft = d_hardening_plastic_theta_d_k_weft(I);
                        Float dq_dh_weft =
                            d_hardening_plastic_theta_d_hardening_k_weft(I);

                        Float dgamma_dy_weft =
                            r_weft * (-s_weft * dp_dy_weft - (1.0 + (hk_weft / k_weft) * dq_dy_weft));
                        Float dgamma_dk_weft =
                            r_weft
                                * (-s_weft * dp_dk_weft
                                   - ((hk_weft / k_weft) * dq_dk_weft
                                      - hardening_plastic_theta_weft * hk_weft
                                            / (k_weft * k_weft)))
                            + f_weft * hk_weft / denom_sq_weft;
                        Float dgamma_dh_weft =
                            r_weft
                                * (-s_weft * dp_dh_weft
                                   - ((hk_weft / k_weft) * dq_dh_weft
                                      + hardening_plastic_theta_weft / k_weft))
                            - f_weft * k_weft / denom_sq_weft;

                        fem_d_plastic_theta_d_yield_ori_weft(I) =
                            dp_dy_weft + s_weft * dgamma_dy_weft;
                        fem_d_plastic_theta_d_k_weft(I) = dp_dk_weft + s_weft * dgamma_dk_weft;
                        fem_d_plastic_theta_d_hardening_k_weft(I) =
                            dp_dh_weft + s_weft * dgamma_dh_weft;
                        d_hardening_plastic_theta_d_yield_ori_weft(I) =
                            dq_dy_weft + dgamma_dy_weft;
                        d_hardening_plastic_theta_d_k_weft(I) = dq_dk_weft + dgamma_dk_weft;
                        d_hardening_plastic_theta_d_hardening_k_weft(I) =
                            dq_dh_weft + dgamma_dh_weft;
                    }

                    //printf("Forward fem_d_plastic_theta_d_yield_ori_weft: %f\r\n",
                    //       fem_d_plastic_theta_d_yield_ori_weft(I));

                    yield_thetas_warp(I) = yield_ori_theta_warp_eff
                        + hardening_plastic_thetas_warp(I) * (hk_warp / k_warp);
                    yield_thetas_weft(I) = yield_ori_theta_weft_eff
                        + hardening_plastic_thetas_weft(I) * (hk_weft / k_weft);

                    Float var_theta_warp = theta_warp - anchor_theta_warp;
                    Float var_theta_weft = theta_weft - anchor_theta_weft;

                    //bool is_slide_warp = abs(var_theta_warp) > thres_theta_warp;
                    //bool is_slide_weft = abs(var_theta_weft) > thres_theta_weft;

                    bool is_slide_warp = abs(var_theta_warp) > thres_theta_warp_eff;
                    bool is_slide_weft = abs(var_theta_weft) > thres_theta_weft_eff;

                    if(is_slide_warp)
                    {
                        //anchor_thetas_warp(I) =
                        //    anchor_theta_warp
                        //    + sign(var_theta_warp) * (abs(var_theta_warp) - thres_theta_warp);

                        anchor_thetas_warp(I) =
                            anchor_theta_warp
                            + sign(var_theta_warp)
                                  * (abs(var_theta_warp) - thres_theta_warp_eff);

                        fem_d_anchor_theta_d_thres_theta_warp(I) = -sign(var_theta_warp);
                    }



                    if(is_slide_weft)
                    {
                        //anchor_thetas_weft(I) =
                        //    anchor_theta_weft
                        //    + sign(var_theta_weft) * (abs(var_theta_weft) - thres_theta_weft);

                        anchor_thetas_weft(I) =
                            anchor_theta_weft
                            + sign(var_theta_weft)
                                  * (abs(var_theta_weft) - thres_theta_weft_eff);

                        fem_d_anchor_theta_d_thres_theta_weft(I) = -sign(var_theta_weft);
                    }

                    //fem_dihedral_thetas(I) = theta;

                    fem_plastic_thetas_warp(I) = plastic_thetas_warp(I);
                    fem_anchor_thetas_warp(I)  = anchor_thetas_warp(I);

                    fem_plastic_thetas_weft(I) = plastic_thetas_weft(I);
                    fem_anchor_thetas_weft(I)  = anchor_thetas_weft(I);

                    // Publish every frame, including inactive gates; do not feed replay into forward.
                    recorded_d_plastic_theta_d_yield_ori_warp(I) =
                        fem_d_plastic_theta_d_yield_ori_warp(I);
                    recorded_d_plastic_theta_d_k_warp(I) =
                        fem_d_plastic_theta_d_k_warp(I);
                    recorded_d_plastic_theta_d_hardening_k_warp(I) =
                        fem_d_plastic_theta_d_hardening_k_warp(I);
                    recorded_d_anchor_theta_d_thres_theta_warp(I) =
                        fem_d_anchor_theta_d_thres_theta_warp(I);
                    recorded_d_plastic_theta_d_yield_ori_weft(I) =
                        fem_d_plastic_theta_d_yield_ori_weft(I);
                    recorded_d_plastic_theta_d_k_weft(I) =
                        fem_d_plastic_theta_d_k_weft(I);
                    recorded_d_plastic_theta_d_hardening_k_weft(I) =
                        fem_d_plastic_theta_d_hardening_k_weft(I);
                    recorded_d_anchor_theta_d_thres_theta_weft(I) =
                        fem_d_anchor_theta_d_thres_theta_weft(I);
                });

        //std::cout << "friction plastic update end" << std::endl;

        dihedral_thetas.view().copy_to(h_dihedral_thetas.data());
        anchor_thetas_warp.view().copy_to(h_anchor_thetas_warp.data());
        anchor_thetas_weft.view().copy_to(h_anchor_thetas_weft.data());
        plastic_thetas_warp.view().copy_to(h_plastic_thetas_warp.data());
        plastic_thetas_weft.view().copy_to(h_plastic_thetas_weft.data());
        hardening_plastic_thetes_warp.view().copy_to(
            h_hardening_plastic_thetas_warp.data());
        hardening_plastic_thetes_weft.view().copy_to(
            h_hardening_plastic_thetas_weft.data());
        yield_thetas_warp.view().copy_to(h_yield_thetas_warp.data());
        yield_thetas_weft.view().copy_to(h_yield_thetas_weft.data());

        auto dihedral_theta_view = view(*geo->edges().find<Float>("dihedral_theta"));
        auto plastic_theta_warp_view =
            view(*geo->edges().find<Float>("plastic_theta_warp"));
        auto plastic_theta_weft_view =
            view(*geo->edges().find<Float>("plastic_theta_weft"));
        auto anchor_theta_warp_view =
            view(*geo->edges().find<Float>("anchor_theta_warp"));
        auto anchor_theta_weft_view =
            view(*geo->edges().find<Float>("anchor_theta_weft"));
        auto yield_theta_warp_view = view(*geo->edges().find<Float>("yield_theta_warp"));
        auto yield_theta_weft_view = view(*geo->edges().find<Float>("yield_theta_weft"));
        auto hardening_plastic_theta_warp_view =
            view(*geo->edges().find<Float>("hardening_plastic_theta_warp"));
        auto hardening_plastic_theta_weft_view =
            view(*geo->edges().find<Float>("hardening_plastic_theta_weft"));

        auto dihedral_theta_view_it     = dihedral_theta_view.begin();
        auto plastic_theta_warp_view_it = plastic_theta_warp_view.begin();
        auto plastic_theta_weft_view_it = plastic_theta_weft_view.begin();
        auto anchor_theta_warp_view_it  = anchor_theta_warp_view.begin();
        auto anchor_theta_weft_view_it  = anchor_theta_weft_view.begin();
        auto yield_theta_warp_view_it   = yield_theta_warp_view.begin();
        auto yield_theta_weft_view_it   = yield_theta_weft_view.begin();
        auto hardening_plastic_theta_warp_view_it =
            hardening_plastic_theta_warp_view.begin();
        auto hardening_plastic_theta_weft_view_it =
            hardening_plastic_theta_weft_view.begin();

        int cnt{0};

        for(int i = 0; i < is_valid_bending.size(); ++i)
        {
            if(is_valid_bending.at(i))
            {
                *dihedral_theta_view_it     = h_dihedral_thetas[cnt];
                *plastic_theta_warp_view_it = h_plastic_thetas_warp[cnt];
                *plastic_theta_weft_view_it = h_plastic_thetas_weft[cnt];
                *anchor_theta_warp_view_it  = h_anchor_thetas_warp[cnt];
                *anchor_theta_weft_view_it  = h_anchor_thetas_weft[cnt];
                *yield_theta_warp_view_it   = h_yield_thetas_warp[cnt];
                *yield_theta_weft_view_it   = h_yield_thetas_weft[cnt];
                *hardening_plastic_theta_warp_view_it =
                    h_hardening_plastic_thetas_warp[cnt];
                *hardening_plastic_theta_weft_view_it =
                    h_hardening_plastic_thetas_weft[cnt];
                cnt++;
            }

            dihedral_theta_view_it++;
            plastic_theta_warp_view_it++;
            plastic_theta_weft_view_it++;
            anchor_theta_warp_view_it++;
            anchor_theta_weft_view_it++;
            yield_theta_warp_view_it++;
            yield_theta_weft_view_it++;
            hardening_plastic_theta_warp_view_it++;
            hardening_plastic_theta_weft_view_it++;
        }
    }

    virtual void do_build(BuildInfo& info) override
    {
        fem                 = &require<FiniteElementMethod>();
        audit_linear_system = &require<GlobalLinearSystem>();
        history_manager     = find<GlobalDiffSimManager>();
        history_position_gradient_enabled = world().scene().info()["diff_sim"].value("orthotropic_bending_history_gradient", true);
        std::cout << "[ortho-history-gradient-v1] history-position adjoint="
                  << (history_position_gradient_enabled ? "ON" : "OFF (approximate)")
                  << "; physical history and parameter sensitivities retained" << std::endl;

        auto& dof_predictor = require<DofPredictor>();

        dof_predictor.on_predict(*this,
                                 [&](DofPredictor::PredictInfo& info)
                                 { friction_plastic_update(); });  // Update Plastic Theta
    }

    virtual void do_init(FilteredInfo& info) override
    {
        std::cout << "Friction Hardening Plastic Orthotropic Bending Model Init"
                  << std::endl;

        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        using ForEachInfo = FiniteElementMethod::ForEachInfo;
        geo_slots         = world().scene().geometries();

        //std::cout << "Number of geo slots size: " << geo_slots.size() << std::endl;

        auto geo = geo_slots.front()->geometry().as<geometry::SimplicialComplex>();
        auto e_count = geo->edges().size();

        for(auto geo_if : info.geo_infos())
        {
            filter_geo_slot_indices.push_back(geo_if.geo_slot_index);
        }

        list<Vector4i> stencil_list;
        list<Float>    bias_angles_list;
        list<Float>    bending_stiffness_warp_list;
        list<Float>    bending_stiffness_weft_list;
        list<Float>    bending_torsion_list;
        list<Float>    bending_poisson_list;

        list<Float> hardening_k_warp_list;
        list<Float> hardening_k_weft_list;

        list<Float> dihedral_theta_list;
        list<Float> plastic_theta_warp_list;
        list<Float> plastic_theta_weft_list;
        list<Float> yield_theta_warp_list;
        list<Float> yield_theta_weft_list;

        list<Float> friction_stiffness_warp_list;
        list<Float> friction_stiffness_weft_list;
        list<Float> thres_theta_warp_list;
        list<Float> thres_theta_weft_list;
        list<Float> anchor_theta_warp_list;
        list<Float> anchor_theta_weft_list;

        // 1) Retrieve Quad Stencils
        info.for_each(  //
            geo_slots,
            [&](const ForEachInfo& I, geometry::SimplicialComplex& sc)
            {
                auto vertex_offset =
                    sc.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
                UIPC_ASSERT(vertex_offset, "Vertex offset not found, why?");
                auto vertex_offset_v = vertex_offset->view().front();

                auto edges = sc.edges().topo().view();

                for(auto&& [i, e] : enumerate(edges))
                {
                    Vector2i E = e;
                    std::sort(E.begin(), E.end());

                    stencil_map[E].edge_index = i;
                }

                auto triangles = sc.triangles().topo().view();
                for(auto&& t : triangles)
                {
                    Vector3i T = t;
                    std::sort(T.begin(), T.end());

                    Vector2i E01 = {T[0], T[1]};
                    Vector2i E02 = {T[0], T[2]};
                    Vector2i E12 = {T[1], T[2]};

                    // insert opposite vertices
                    stencil_map[E01].oppo_verts.insert(T[2]);
                    stencil_map[E02].oppo_verts.insert(T[1]);
                    stencil_map[E12].oppo_verts.insert(T[0]);
                }

                auto bending_stiffnesses_warp =
                    sc.edges().find<Float>("bending_stiffness_warp");
                UIPC_ASSERT(bending_stiffnesses_warp, "Bending stiffness (warp) not found, why?");
                auto bs_warp_view = bending_stiffnesses_warp->view();

                auto bending_stiffnesses_weft =
                    sc.edges().find<Float>("bending_stiffness_weft");
                UIPC_ASSERT(bending_stiffnesses_weft, "Bending stiffness (weft) not found, why?");
                auto bs_weft_view = bending_stiffnesses_weft->view();

                auto bending_torsion = sc.edges().find<Float>("bending_torsion");
                UIPC_ASSERT(bending_torsion, "Bending torsion not found, why?");
                auto bs_torsion_view = bending_torsion->view();

                auto bending_poisson = sc.edges().find<Float>("bending_poisson");
                UIPC_ASSERT(bending_poisson, "Bending poisson not found, why?");
                auto bs_poisson_view = bending_poisson->view();

                auto hardening_ks_warp = sc.edges().find<Float>("hardening_k_warp");
                UIPC_ASSERT(hardening_ks_warp, "Hardening_k (warp) not found, why?");
                auto hks_warp_view = hardening_ks_warp->view();

                auto hardening_ks_weft = sc.edges().find<Float>("hardening_k_weft");
                UIPC_ASSERT(hardening_ks_weft, "Hardening_k (weft) not found, why?");
                auto hks_weft_view = hardening_ks_weft->view();

                auto dihedral_theta = sc.edges().find<Float>("dihedral_theta");
                UIPC_ASSERT(dihedral_theta, "Dihedral Theta not found, why?");
                auto ds_view = dihedral_theta->view();

                // Plastic Properties
                auto plastic_thetas_warp = sc.edges().find<Float>("plastic_theta_warp");
                UIPC_ASSERT(plastic_thetas_warp, "Plastic theta (warp) is not found!");
                auto pt_warp_view = plastic_thetas_warp->view();

                auto plastic_thetas_weft = sc.edges().find<Float>("plastic_theta_weft");
                UIPC_ASSERT(plastic_thetas_weft, "Plastic theta (weft) is not found!");
                auto pt_weft_view = plastic_thetas_weft->view();

                auto hardening_plastic_theta_warp =
                    sc.edges().find<Float>("hardening_plastic_theta_warp");
                if(!hardening_plastic_theta_warp)
                    hardening_plastic_theta_warp =
                        sc.edges().create<Float>("hardening_plastic_theta_warp", 0.0);
                std::ranges::fill(geometry::view(*hardening_plastic_theta_warp), 0.0);

                auto hardening_plastic_theta_weft =
                    sc.edges().find<Float>("hardening_plastic_theta_weft");
                if(!hardening_plastic_theta_weft)
                    hardening_plastic_theta_weft =
                        sc.edges().create<Float>("hardening_plastic_theta_weft", 0.0);
                std::ranges::fill(geometry::view(*hardening_plastic_theta_weft), 0.0);

                auto yield_thetas_warp = sc.edges().find<Float>("yield_theta_warp");
                UIPC_ASSERT(yield_thetas_warp, "Yield theta (warp) is not found!");
                auto yt_warp_view = yield_thetas_warp->view();

                auto yield_thetas_weft = sc.edges().find<Float>("yield_theta_weft");
                UIPC_ASSERT(yield_thetas_weft, "Yield theta (weft) is not found!");
                auto yt_weft_view = yield_thetas_weft->view();

                // Find Friction Properties
                auto friction_stiffnesses_warp =
                    sc.edges().find<Float>("friction_stiffness_warp");
                UIPC_ASSERT(friction_stiffnesses_warp,
                            "Friction Stiffness (warp) not found, why?");
                auto fs_warp_view = friction_stiffnesses_warp->view();

                auto friction_stiffnesses_weft =
                    sc.edges().find<Float>("friction_stiffness_weft");
                UIPC_ASSERT(friction_stiffnesses_weft,
                            "Friction Stiffness (weft) not found, why?");
                auto fs_weft_view = friction_stiffnesses_weft->view();

                auto thres_thetas_warp = sc.edges().find<Float>("thres_theta_warp");
                UIPC_ASSERT(thres_thetas_warp, "Threshold Theta (warp) not found, why?");
                auto tt_warp_view = thres_thetas_warp->view();

                auto thres_thetas_weft = sc.edges().find<Float>("thres_theta_weft");
                UIPC_ASSERT(thres_thetas_weft, "Threshold Theta (weft) not found, why?");
                auto tt_weft_view = thres_thetas_weft->view();

                auto anchor_thetas_warp = sc.edges().find<Float>("anchor_theta_warp");
                UIPC_ASSERT(anchor_thetas_warp, "Anchor Theta (warp) not found, why?");
                auto at_warp_view = anchor_thetas_warp->view();

                auto anchor_thetas_weft = sc.edges().find<Float>("anchor_theta_weft");
                UIPC_ASSERT(anchor_thetas_weft, "Anchor Theta (weft) not found, why?");
                auto at_weft_view = anchor_thetas_weft->view();

                auto reset_anchor_plastic = sc.edges().find<I32>("reset_anchor_plastic");
                if(!reset_anchor_plastic)
                    reset_anchor_plastic =
                        sc.edges().create<I32>("reset_anchor_plastic", 0);
                std::ranges::fill(geometry::view(*reset_anchor_plastic), 0);

                // Create Const Geometry Attributes
                auto l0s = sc.edges().find<Float>("L0");
                if(!l0s)
                    l0s = sc.edges().create<Float>("L0");
                auto l0s_view = geometry::view(*l0s);
                std::ranges::fill(l0s_view, 0.0);

                auto v_bars = sc.edges().find<Float>("V_bar");
                if(!v_bars)
                    v_bars = sc.edges().create<Float>("V_bar");
                auto v_bars_view = geometry::view(*v_bars);
                std::ranges::fill(v_bars_view, 0.0);

                auto h_bars = sc.edges().find<Float>("h_bar");
                if(!h_bars)
                    h_bars = sc.edges().create<Float>("h_bar");
                auto h_bars_view = geometry::view(*h_bars);
                std::ranges::fill(h_bars_view, 0.0);

                auto torsion_bars = sc.edges().find<Float>("torsion_bars");
                if(!torsion_bars)
                    torsion_bars = sc.edges().create<Float>("torsion_bars");
                auto torsion_bars_view = geometry::view(*torsion_bars);
                std::ranges::fill(torsion_bars_view, 0.0);

                // Bias Angle of Each Bending Edge
                auto bias_angles_attr = sc.edges().find<Float>("bias_angle");
                UIPC_ASSERT(bias_angles_attr, "bias_angle not found, why?");
                auto bias_angles_view = bias_angles_attr->view();

                auto stencils = sc.edges().find<Vector4i>("stencil");
                if(!stencils)
                    stencils = sc.edges().create<Vector4i>("stencil");
                auto stencils_view = geometry::view(*stencils);
                std::ranges::fill(stencils_view, Vector4i::Zero());

                //int cnt_bias_angle{0};

                for(auto&& [E, info] : stencil_map)
                {
                    if(info.valid_bending())
                    {
                        // X0, X1, X2, X3; (X1, X2) is middle edge
                        Vector4i stencil{*info.oppo_verts.begin(),    // X0
                                         E(0),                        // X1
                                         E(1),                        // X2
                                         *info.oppo_verts.rbegin()};  // X3

                        auto stencil_index = stencil_list.size();
                        info.stencil_index = stencil_index;  // save stencil index

                        // convert to fem vertex index
                        stencil_list.push_back(stencil.array() + vertex_offset_v);

                        Float bs_warp = bs_warp_view[info.edge_index];
                        bending_stiffness_warp_list.push_back(bs_warp);

                        Float bs_weft = bs_weft_view[info.edge_index];
                        bending_stiffness_weft_list.push_back(bs_weft);

                        Float bs_torsion = bs_torsion_view[info.edge_index];
                        bending_torsion_list.push_back(bs_torsion);

                        Float bs_poisson = bs_poisson_view[info.edge_index];
                        bending_poisson_list.push_back(bs_poisson);

                        Float hk_warp = hks_warp_view[info.edge_index];
                        hardening_k_warp_list.push_back(hk_warp);

                        Float hk_weft = hks_weft_view[info.edge_index];
                        hardening_k_weft_list.push_back(hk_weft);

                        Float ds = ds_view[info.edge_index];
                        dihedral_theta_list.push_back(ds);

                        Float pt_warp = pt_warp_view[info.edge_index];
                        plastic_theta_warp_list.push_back(pt_warp);

                        Float pt_weft = pt_weft_view[info.edge_index];
                        plastic_theta_weft_list.push_back(pt_weft);

                        Float yt_warp = yt_warp_view[info.edge_index];
                        yield_theta_warp_list.push_back(yt_warp);

                        Float yt_weft = yt_weft_view[info.edge_index];
                        yield_theta_weft_list.push_back(yt_weft);

                        // Friction
                        Float fs_warp = fs_warp_view[info.edge_index];
                        friction_stiffness_warp_list.push_back(fs_warp);

                        Float fs_weft = fs_weft_view[info.edge_index];
                        friction_stiffness_weft_list.push_back(fs_weft);

                        Float tt_warp = tt_warp_view[info.edge_index];
                        thres_theta_warp_list.push_back(tt_warp);

                        Float tt_weft = tt_weft_view[info.edge_index];
                        thres_theta_weft_list.push_back(tt_weft);

                        Float acht_warp = at_warp_view[info.edge_index];
                        anchor_theta_warp_list.push_back(acht_warp);

                        Float acht_weft = at_weft_view[info.edge_index];
                        anchor_theta_weft_list.push_back(acht_weft);

                        //std::cout << "Valid Edge Index: " << info.edge_index << std::endl;

                        //std::cout << "Stencile: " << std::endl;

                        //std::cout << stencil << std::endl;

                        Float bias_angle = bias_angles_view[info.edge_index];
                        bias_angles_list.push_back(bias_angle);

                        //Float bias_angle = bias_angles_view[cnt_bias_angle];
                        //bias_angles_list.push_back(bias_angle);

                        //std::cout << "Bias Angle: " << bias_angle << std::endl;

                        is_valid_bending.push_back(true);
                    }
                    else
                    {
                        is_valid_bending.push_back(false);
                    }

                    //cnt_bias_angle++;
                }
            });

        //std::cout << "After Copy to List" << std::endl;

        // 2) Setup Invariant Data
        h_stencils.resize(stencil_list.size());
        h_bending_stiffness_warp.resize(stencil_list.size());
        h_bending_stiffness_weft.resize(stencil_list.size());
        h_bending_torsion.resize(stencil_list.size());
        h_bending_poisson.resize(stencil_list.size());

        std::ranges::move(stencil_list, h_stencils.begin());
        std::ranges::move(bending_stiffness_warp_list, h_bending_stiffness_warp.begin());
        std::ranges::move(bending_stiffness_weft_list, h_bending_stiffness_weft.begin());
        std::ranges::move(bending_torsion_list, h_bending_torsion.begin());
        std::ranges::move(bending_poisson_list, h_bending_poisson.begin());

        h_bias_angles.resize(h_stencils.size());
        std::ranges::move(bias_angles_list, h_bias_angles.begin());

        //      2.1): Plastic Parameters
        h_hardening_ks_warp.resize(stencil_list.size());
        h_hardening_ks_weft.resize(stencil_list.size());
        h_yield_thetas_warp.resize(stencil_list.size());
        h_yield_thetas_weft.resize(stencil_list.size());
        h_plastic_thetas_warp.resize(stencil_list.size());
        h_plastic_thetas_weft.resize(stencil_list.size());
        h_dihedral_thetas.resize(stencil_list.size());


        std::ranges::move(hardening_k_warp_list, h_hardening_ks_warp.begin());
        std::ranges::move(hardening_k_weft_list, h_hardening_ks_weft.begin());
        std::ranges::move(yield_theta_warp_list, h_yield_thetas_warp.begin());
        std::ranges::move(yield_theta_weft_list, h_yield_thetas_weft.begin());
        std::ranges::move(plastic_theta_warp_list, h_plastic_thetas_warp.begin());
        std::ranges::move(plastic_theta_weft_list, h_plastic_thetas_weft.begin());
        std::ranges::move(dihedral_theta_list, h_dihedral_thetas.begin());
        //      2.2): Friction Parameters
        h_friction_stiffness_warp.resize(stencil_list.size());
        h_friction_stiffness_weft.resize(stencil_list.size());
        h_anchor_thetas_warp.resize(stencil_list.size());
        h_anchor_thetas_weft.resize(stencil_list.size());
        h_thres_thetas_warp.resize(stencil_list.size());
        h_thres_thetas_weft.resize(stencil_list.size());
        std::ranges::move(friction_stiffness_warp_list,
                          h_friction_stiffness_warp.begin());
        std::ranges::move(friction_stiffness_weft_list,
                          h_friction_stiffness_weft.begin());
        std::ranges::move(anchor_theta_warp_list, h_anchor_thetas_warp.begin());
        std::ranges::move(anchor_theta_weft_list, h_anchor_thetas_weft.begin());
        std::ranges::move(thres_theta_warp_list, h_thres_thetas_warp.begin());
        std::ranges::move(thres_theta_weft_list, h_thres_thetas_weft.begin());

        // 3) Setup Related Data
        span x_bars      = info.rest_positions();
        span thicknesses = info.thicknesses();
        h_rest_lengths.resize(h_stencils.size());
        h_h_bars.resize(h_stencils.size());
        h_V_bars.resize(h_stencils.size());
        h_theta_bars.resize(h_stencils.size());
        h_theta_torsion_bar.resize(h_stencils.size());

        for(auto&& [i, stencil] : enumerate(h_stencils))
        {
            Vector3 X0         = x_bars[stencil[0]];
            Vector3 X1         = x_bars[stencil[1]];
            Vector3 X2         = x_bars[stencil[2]];
            Vector3 X3         = x_bars[stencil[3]];
            Float   thickness0 = thicknesses[stencil[0]];
            Float   thickness1 = thicknesses[stencil[1]];
            Float   thickness2 = thicknesses[stencil[2]];
            Float   thickness3 = thicknesses[stencil[3]];

            Float bias_ang = h_bias_angles[i];

            //std::cout << "Edge Index: " << i << std::endl;

            //std::cout << "Stenciel :" << stencil[0] << ", " << stencil[1]
            //          << ", " << stencil[2] << ", " << stencil[3] << std::endl;

            //std::cout << "x0: " << X0 << std::endl;
            //std::cout << "x1: " << X1 << std::endl;
            //std::cout << "x2: " << X2 << std::endl;
            //std::cout << "x3: " << X3 << std::endl;

            Float angle_between_warp = bias_ang + 1.570796;
            Float cos_angle          = cos(angle_between_warp);
            Float sin_angle          = sin(angle_between_warp);

            //std::cout << "Bias angle: " << bias_ang << std::endl;

            //std::cout << "angle_between_warp: " << angle_between_warp << std::endl;

            //std::cout << "cos angle: " << cos_angle
            //          << "sin angle: " << sin_angle << std::endl;

            Float L0, V_bar, h_bar, theta_bar;
            DSGFPB::compute_constants(L0,
                                      h_bar,
                                      theta_bar,
                                      V_bar,  //
                                      X0,
                                      X1,
                                      X2,
                                      X3,  //
                                      thickness0,
                                      thickness1,
                                      thickness2,
                                      thickness3);

            h_dihedral_thetas[i] = theta_bar;  // Current dihedral thetas

            Float theta_warp    = theta_bar * cos_angle * cos_angle;
            Float theta_weft    = theta_bar * sin_angle * sin_angle;
            Float theta_torsion = theta_bar * cos_angle * sin_angle;

            //std::cout << "theta warp: " << theta_warp << std::endl;
            //std::cout << "theta weft: " << theta_weft << std::endl;
            //std::cout << "theta torsion: " << theta_torsion << std::endl;

            h_theta_torsion_bar[i] = theta_torsion;

            h_plastic_thetas_warp[i] = theta_warp;
            h_plastic_thetas_weft[i] = theta_weft;

            h_anchor_thetas_warp[i] = theta_warp;
            h_anchor_thetas_weft[i] = theta_weft;

            h_rest_lengths[i] = L0;
            h_h_bars[i]       = h_bar;
            h_theta_bars[i]   = 0.0;
            h_V_bars[i]       = V_bar;
        }


        //Used when simulate single object
        //Initialize the theta angles.
        auto dihedral_theta_view = view(*geo->edges().find<Float>("dihedral_theta"));

        auto plastic_theta_warp_view =
            view(*geo->edges().find<Float>("plastic_theta_warp"));
        auto plastic_theta_weft_view =
            view(*geo->edges().find<Float>("plastic_theta_weft"));
        auto anchor_theta_warp_view =
            view(*geo->edges().find<Float>("anchor_theta_warp"));
        auto anchor_theta_weft_view =
            view(*geo->edges().find<Float>("anchor_theta_weft"));

        auto dihedral_theta_view_it     = dihedral_theta_view.begin();
        auto plastic_theta_warp_view_it = plastic_theta_warp_view.begin();
        auto plastic_theta_weft_view_it = plastic_theta_weft_view.begin();
        auto anchor_theta_warp_view_it  = anchor_theta_warp_view.begin();
        auto anchor_theta_weft_view_it  = anchor_theta_weft_view.begin();

        auto L0s_view    = view(*geo->edges().find<Float>("L0"));
        auto V_bars_view = view(*geo->edges().find<Float>("V_bar"));
        auto h_bars_view = view(*geo->edges().find<Float>("h_bar"));
        auto torsion_bars_view = view(*geo->edges().find<Float>("torsion_bars"));
        auto stencils_view = view(*geo->edges().find<Vector4i>("stencil"));

        auto L0s_view_it          = L0s_view.begin();
        auto V_bars_view_it       = V_bars_view.begin();
        auto h_bars_view_it       = h_bars_view.begin();
        auto torsion_bars_view_it = torsion_bars_view.begin();
        auto stencils_view_it     = stencils_view.begin();

        int cnt{0};

        for(int i = 0; i < is_valid_bending.size(); ++i)
        {
            if(is_valid_bending.at(i))
            {
                *dihedral_theta_view_it     = h_dihedral_thetas[cnt];
                *plastic_theta_warp_view_it = h_plastic_thetas_warp[cnt];
                *plastic_theta_weft_view_it = h_plastic_thetas_weft[cnt];
                *anchor_theta_warp_view_it  = h_anchor_thetas_warp[cnt];
                *anchor_theta_weft_view_it  = h_anchor_thetas_weft[cnt];

                *L0s_view_it          = h_rest_lengths[cnt];
                *V_bars_view_it       = h_V_bars[cnt];
                *h_bars_view_it       = h_h_bars[cnt];
                *torsion_bars_view_it = h_theta_torsion_bar[cnt];
                *stencils_view_it     = h_stencils[cnt];
                cnt++;
            }

            dihedral_theta_view_it++;
            plastic_theta_warp_view_it++;
            plastic_theta_weft_view_it++;
            anchor_theta_warp_view_it++;
            anchor_theta_weft_view_it++;

            L0s_view_it++;
            V_bars_view_it++;
            h_bars_view_it++;
            torsion_bars_view_it++;
            stencils_view_it++;
        }

        //// Used when simulate multiple objects
        //int cnt{0};

        //info.for_each(
        //    geo_slots,
        //    [&](ForEachInfo& I, geometry::SimplicialComplex& geo)
        //    {
        //        auto dihedral_theta_view =
        //            view(*geo.edges().find<Float>("dihedral_theta"));
        //        auto plastic_theta_view = view(*geo.edges().find<Float>("plastic_theta"));
        //        auto anchor_theta_view = view(*geo.edges().find<Float>("anchor_theta"));

        //        auto dihedral_theta_view_it = dihedral_theta_view.begin();
        //        auto plastic_theta_view_it  = plastic_theta_view.begin();
        //        auto anchor_theta_view_it   = anchor_theta_view.begin();

        //        auto L0s_view    = view(*geo.edges().find<Float>("L0"));
        //        auto V_bars_view = view(*geo.edges().find<Float>("V_bar"));
        //        auto h_bars_view = view(*geo.edges().find<Float>("h_bar"));
        //        auto stencils_view = view(*geo.edges().find<Vector4i>("stencil"));

        //        auto L0s_view_it      = L0s_view.begin();
        //        auto V_bars_view_it   = V_bars_view.begin();
        //        auto h_bars_view_it   = h_bars_view.begin();
        //        auto stencils_view_it = stencils_view.begin();

        //        for(int i = 0; i < is_valid_bending.size(); ++i)
        //        {
        //            if(is_valid_bending.at(i))
        //            {
        //                *dihedral_theta_view_it = h_dihedral_thetas[cnt];
        //                *plastic_theta_view_it  = h_dihedral_thetas[cnt];
        //                *anchor_theta_view_it   = h_dihedral_thetas[cnt];

        //                *L0s_view_it      = h_rest_lengths[cnt];
        //                *V_bars_view_it   = h_V_bars[cnt];
        //                *h_bars_view_it   = h_h_bars[cnt];
        //                *stencils_view_it = h_stencils[cnt];

        //                cnt++;
        //            }

        //            dihedral_theta_view_it++;
        //            plastic_theta_view_it++;
        //            anchor_theta_view_it++;

        //            L0s_view_it++;
        //            V_bars_view_it++;
        //            h_bars_view_it++;
        //            stencils_view_it++;
        //        }
        //    });

        // 4) Copy to Device
        stencils.resize(h_stencils.size());
        stencils.view().copy_from(h_stencils.data());

        bending_stiffnesses_warp.resize(h_bending_stiffness_warp.size());
        bending_stiffnesses_warp.view().copy_from(h_bending_stiffness_warp.data());

        bending_stiffnesses_weft.resize(h_bending_stiffness_weft.size());
        bending_stiffnesses_weft.view().copy_from(h_bending_stiffness_weft.data());

        bending_torsions.resize(h_bending_torsion.size());
        bending_torsions.view().copy_from(h_bending_torsion.data());

        bending_poissons.resize(h_bending_poisson.size());
        bending_poissons.view().copy_from(h_bending_poisson.data());

        hardening_ks_warp.resize(h_hardening_ks_warp.size());
        hardening_ks_warp.view().copy_from(h_hardening_ks_warp.data());

        hardening_ks_weft.resize(h_hardening_ks_weft.size());
        hardening_ks_weft.view().copy_from(h_hardening_ks_weft.data());

        dihedral_thetas.resize(h_dihedral_thetas.size());
        dihedral_thetas.view().copy_from(h_dihedral_thetas.data());

        theta_torsion_bars.resize(h_theta_torsion_bar.size());
        theta_torsion_bars.view().copy_from(h_theta_torsion_bar.data());

        // Plastic Properties
        hardening_plastic_thetes_warp.resize(h_hardening_ks_warp.size());
        hardening_plastic_thetes_warp.fill(0.0);

        h_hardening_plastic_thetas_warp.resize(h_hardening_ks_warp.size());
        hardening_plastic_thetes_warp.view().copy_to(
            h_hardening_plastic_thetas_warp.data());

        hardening_plastic_thetes_weft.resize(h_hardening_ks_weft.size());
        hardening_plastic_thetes_weft.fill(0.0);

        h_hardening_plastic_thetas_weft.resize(h_hardening_ks_weft.size());
        hardening_plastic_thetes_weft.view().copy_to(
            h_hardening_plastic_thetas_weft.data());

        d_hardening_plastic_theta_d_yield_ori_warp.resize(h_hardening_ks_warp.size());
        d_hardening_plastic_theta_d_yield_ori_warp.fill(0.0);
        d_hardening_plastic_theta_d_yield_ori_weft.resize(h_hardening_ks_weft.size());
        d_hardening_plastic_theta_d_yield_ori_weft.fill(0.0);
        d_hardening_plastic_theta_d_k_warp.resize(h_hardening_ks_warp.size());
        d_hardening_plastic_theta_d_k_warp.fill(0.0);
        d_hardening_plastic_theta_d_k_weft.resize(h_hardening_ks_weft.size());
        d_hardening_plastic_theta_d_k_weft.fill(0.0);
        d_hardening_plastic_theta_d_hardening_k_warp.resize(h_hardening_ks_warp.size());
        d_hardening_plastic_theta_d_hardening_k_warp.fill(0.0);
        d_hardening_plastic_theta_d_hardening_k_weft.resize(h_hardening_ks_weft.size());
        d_hardening_plastic_theta_d_hardening_k_weft.fill(0.0);

        yield_thetas_warp.resize(h_yield_thetas_warp.size());
        yield_thetas_warp.view().copy_from(h_yield_thetas_warp.data());

        yield_thetas_weft.resize(h_yield_thetas_weft.size());
        yield_thetas_weft.view().copy_from(h_yield_thetas_weft.data());

        yield_ori_thetas_warp.resize(h_yield_thetas_warp.size());
        yield_ori_thetas_warp.view().copy_from(h_yield_thetas_warp.data());

        yield_ori_thetas_weft.resize(h_yield_thetas_weft.size());
        yield_ori_thetas_weft.view().copy_from(h_yield_thetas_weft.data());

        plastic_thetas_warp.resize(h_plastic_thetas_warp.size());
        plastic_thetas_warp.view().copy_from(h_plastic_thetas_warp.data());

        plastic_thetas_weft.resize(h_plastic_thetas_weft.size());
        plastic_thetas_weft.view().copy_from(h_plastic_thetas_weft.data());

        // Friction Properties
        friction_stiffnesses_warp.resize(h_friction_stiffness_warp.size());
        friction_stiffnesses_warp.view().copy_from(h_friction_stiffness_warp.data());

        friction_stiffnesses_weft.resize(h_friction_stiffness_weft.size());
        friction_stiffnesses_weft.view().copy_from(h_friction_stiffness_weft.data());

        thres_thetas_warp.resize(h_thres_thetas_warp.size());
        thres_thetas_warp.view().copy_from(h_thres_thetas_warp.data());

        thres_thetas_weft.resize(h_thres_thetas_weft.size());
        thres_thetas_weft.view().copy_from(h_thres_thetas_weft.data());

        anchor_thetas_warp.resize(h_anchor_thetas_warp.size());
        anchor_thetas_warp.view().copy_from(h_anchor_thetas_warp.data());

        anchor_thetas_weft.resize(h_anchor_thetas_weft.size());
        anchor_thetas_weft.view().copy_from(h_anchor_thetas_weft.data());

        // Elastic: Save for backward
        //fem->dihedral_thetas.resize(dihedral_thetas.size());
        //fem->dihedral_thetas.view().copy_from(h_dihedral_thetas.data());


        forward_d_plastic_theta_d_yield_ori_warp.resize(stencils.size());
        forward_d_plastic_theta_d_yield_ori_warp.fill(0.0);
        forward_d_plastic_theta_d_k_warp.resize(stencils.size());
        forward_d_plastic_theta_d_k_warp.fill(0.0);
        forward_d_plastic_theta_d_hardening_k_warp.resize(stencils.size());
        forward_d_plastic_theta_d_hardening_k_warp.fill(0.0);
        forward_d_anchor_theta_d_thres_theta_warp.resize(stencils.size());
        forward_d_anchor_theta_d_thres_theta_warp.fill(0.0);
        forward_d_plastic_theta_d_yield_ori_weft.resize(stencils.size());
        forward_d_plastic_theta_d_yield_ori_weft.fill(0.0);
        forward_d_plastic_theta_d_k_weft.resize(stencils.size());
        forward_d_plastic_theta_d_k_weft.fill(0.0);
        forward_d_plastic_theta_d_hardening_k_weft.resize(stencils.size());
        forward_d_plastic_theta_d_hardening_k_weft.fill(0.0);
        forward_d_anchor_theta_d_thres_theta_weft.resize(stencils.size());
        forward_d_anchor_theta_d_thres_theta_weft.fill(0.0);

        // Plastic: Save for backward
        fem->plastic_thetas_warp.resize(h_plastic_thetas_warp.size());
        fem->plastic_thetas_warp.view().copy_from(h_plastic_thetas_warp.data());

        fem->plastic_thetas_weft.resize(h_plastic_thetas_weft.size());
        fem->plastic_thetas_weft.view().copy_from(h_plastic_thetas_weft.data());

        fem->d_plastic_theta_d_yield_ori_warp.resize(h_plastic_thetas_warp.size());
        fem->d_plastic_theta_d_yield_ori_warp.fill(0.0);

        fem->d_plastic_theta_d_yield_ori_weft.resize(h_plastic_thetas_weft.size());
        fem->d_plastic_theta_d_yield_ori_weft.fill(0.0);

        fem->d_plastic_theta_d_k_warp.resize(h_plastic_thetas_warp.size());
        fem->d_plastic_theta_d_k_warp.fill(0.0);

        fem->d_plastic_theta_d_k_weft.resize(h_plastic_thetas_weft.size());
        fem->d_plastic_theta_d_k_weft.fill(0.0);

        fem->d_plastic_theta_d_hardening_k_warp.resize(h_plastic_thetas_warp.size());
        fem->d_plastic_theta_d_hardening_k_warp.fill(0.0);

        fem->d_plastic_theta_d_hardening_k_weft.resize(h_plastic_thetas_weft.size());
        fem->d_plastic_theta_d_hardening_k_weft.fill(0.0);

        // Friction: Save for backward
        fem->anchor_thetas_warp.resize(h_anchor_thetas_warp.size());
        fem->anchor_thetas_warp.view().copy_from(h_anchor_thetas_warp.data());

        fem->anchor_thetas_weft.resize(h_anchor_thetas_weft.size());
        fem->anchor_thetas_weft.view().copy_from(h_anchor_thetas_weft.data());

        fem->d_anchor_theta_d_thres_theta_warp.resize(h_anchor_thetas_warp.size());
        fem->d_anchor_theta_d_thres_theta_warp.fill(0.0);

        fem->d_anchor_theta_d_thres_theta_weft.resize(h_anchor_thetas_weft.size());
        fem->d_anchor_theta_d_thres_theta_weft.fill(0.0);

        // Constants
        rest_lengths.resize(h_rest_lengths.size());
        rest_lengths.view().copy_from(h_rest_lengths.data());

        h_bars.resize(h_h_bars.size());
        h_bars.view().copy_from(h_h_bars.data());

        theta_bars.resize(h_theta_bars.size());
        theta_bars.view().copy_from(h_theta_bars.data());

        V_bars.resize(h_V_bars.size());
        V_bars.view().copy_from(h_V_bars.data());

        bias_angles.resize(h_bias_angles.size());
        bias_angles.view().copy_from(h_bias_angles.data());
    }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(stencils.size());  // Each quad has 1 energy
        info.stencil_dim(4);                 // Each quad has 4 vertices
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                info.energies().size(),
                [stencils    = stencils.viewer().name("stencils"),
                 bias_angles = bias_angles.cviewer().name("bias_angle"),
                 bending_stiffnesses_warp = bending_stiffnesses_warp.viewer().name("bending_stiffness_warp"),
                 bending_stiffnesses_weft = bending_stiffnesses_weft.viewer().name("bending_stiffness_weft"),
                 bending_torsions = bending_torsions.viewer().name("bending_torsion"),
                 bending_poissons = bending_poissons.viewer().name("bending_poisson"),
                 plastic_thetas_warp = plastic_thetas_warp.viewer().name("plastic_theta_warp"),
                 plastic_thetas_weft = plastic_thetas_weft.viewer().name("plastic_theta_weft"),
                 friction_stiffnesses_warp =
                     friction_stiffnesses_warp.viewer().name("friction_stiffness_warp"),
                 friction_stiffnesses_weft =
                     friction_stiffnesses_weft.viewer().name("friction_stiffness_weft"),
                 anchor_thetas_warp = anchor_thetas_warp.viewer().name("anchor_theta_warp"),
                 anchor_thetas_weft = anchor_thetas_weft.viewer().name("anchor_theta_weft"),
                 h_bars = h_bars.viewer().name("h_bar"),
                 V_bars = V_bars.viewer().name("V_bar"),
                 theta_torsion_bars = theta_torsion_bars.cviewer().name("theta_torsion_bar"),
                 L0s      = rest_lengths.viewer().name("rest_lengths"),
                 xs       = info.xs().viewer().name("xs"),
                 energies = info.energies().viewer().name("energies"),
                 dt       = info.dt()] __device__(int I)
                {
                    Vector4i stencil     = stencils(I);
                    Float    L0          = L0s(I);
                    Float    h_bar       = h_bars(I);
                    Float    torsion_bar = theta_torsion_bars(I);
                    Float    bias_angle  = bias_angles(I);

                    Float kappa_warp    = bending_stiffnesses_warp(I);
                    Float kappa_weft    = bending_stiffnesses_weft(I);
                    Float kappa_torsion = bending_torsions(I);
                    Float kappa_poisson = bending_poissons(I);

                    Float plastic_theta_warp = plastic_thetas_warp(I);
                    Float plastic_theta_weft = plastic_thetas_weft(I);
                    Float friction_stiffness_warp = friction_stiffnesses_warp(I);
                    Float friction_stiffness_weft = friction_stiffnesses_weft(I);
                    Float anchor_theta_warp = anchor_thetas_warp(I);
                    Float anchor_theta_weft = anchor_thetas_weft(I);

                    Float V_bar = V_bars(I);

                    Vector3 x0 = xs(stencil[0]);
                    Vector3 x1 = xs(stencil[1]);
                    Vector3 x2 = xs(stencil[2]);
                    Vector3 x3 = xs(stencil[3]);

                    Float E = DSGFPB::E(x0,
                                        x1,
                                        x2,
                                        x3,
                                        L0,
                                        h_bar,
                                        torsion_bar,
                                        bias_angle,
                                        plastic_theta_warp,
                                        plastic_theta_weft,
                                        anchor_theta_warp,
                                        anchor_theta_weft,
                                        kappa_warp,
                                        kappa_weft,
                                        kappa_torsion,
                                        kappa_poisson,
                                        friction_stiffness_warp,
                                        friction_stiffness_weft);

                    energies(I) = E * V_bar * dt * dt
                        * orthotropic_bending_normalization(bias_angle);
                });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace DSGPB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        // Forward keeps private state; backward must use the state of the replayed frame.
        auto& plastic_warp_state =
            info.is_adjoint() ? fem->plastic_thetas_warp : plastic_thetas_warp;
        auto& plastic_weft_state =
            info.is_adjoint() ? fem->plastic_thetas_weft : plastic_thetas_weft;
        auto& anchor_warp_state = info.is_adjoint() ? fem->anchor_thetas_warp : anchor_thetas_warp;
        auto& anchor_weft_state = info.is_adjoint() ? fem->anchor_thetas_weft : anchor_thetas_weft;
        const bool audit_raw = info.is_adjoint() && audit_linear_system
                               && audit_linear_system->diagnostic_raw_material_hessian();

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                stencils.size(),
                [stencils    = stencils.viewer().name("stencils"),
                 bias_angles = bias_angles.cviewer().name("bias_angle"),
                 bending_stiffnesses_warp = bending_stiffnesses_warp.viewer().name("bending_stiffness_warp"),
                 bending_stiffnesses_weft = bending_stiffnesses_weft.viewer().name("bending_stiffness_weft"),
                 bending_torsions = bending_torsions.viewer().name("bending_torsion"),
                 bending_poissons = bending_poissons.viewer().name("bending_poisson"),
                 plastic_thetas_warp = plastic_warp_state.viewer().name("plastic_theta_warp"),
                 plastic_thetas_weft = plastic_weft_state.viewer().name("plastic_theta_weft"),
                 yield_thetas_warp = yield_thetas_warp.viewer().name("yield_theta_warp"),
                 yield_thetas_weft = yield_thetas_weft.viewer().name("yield_theta_weft"),
                 friction_stiffnesses_warp =
                     friction_stiffnesses_warp.viewer().name("friction_stiffness_warp"),
                 friction_stiffnesses_weft =
                     friction_stiffnesses_weft.viewer().name("friction_stiffness_weft"),
                 anchor_thetas_warp = anchor_warp_state.viewer().name("anchor_theta_warp"),
                 anchor_thetas_weft = anchor_weft_state.viewer().name("anchor_theta_weft"),
                 thres_thetas_warp = thres_thetas_warp.viewer().name("thres_theta_warp"),
                 thres_thetas_weft = thres_thetas_weft.viewer().name("thres_theta_weft"),
                 thicknesses = info.thicknesses().viewer().name("thicknesses"),
                 h_bars      = h_bars.viewer().name("h_bar"),
                 V_bars      = V_bars.viewer().name("V_bar"),
                 theta_torsion_bars = theta_torsion_bars.cviewer().name("theta_torsion_bar"),
                 L0s        = rest_lengths.viewer().name("rest_lengths"),
                 xs         = info.xs().viewer().name("xs"),
                 G3s        = info.gradients().viewer().name("gradients"),
                 H3x3s      = info.hessians().viewer().name("hessians"),
                 dt         = info.dt(),
                 is_adjoint = info.is_adjoint(),
                 audit_raw  = audit_raw] __device__(int I) mutable
                {
                    Vector4i stencil = stencils(I);

                    Float kappa_warp      = bending_stiffnesses_warp(I);
                    Float kappa_weft      = bending_stiffnesses_weft(I);
                    Float bending_torsion = bending_torsions(I);
                    Float bending_poisson = bending_poissons(I);

                    Float friction_stiffness_warp = friction_stiffnesses_warp(I);
                    Float friction_stiffness_weft = friction_stiffnesses_weft(I);

                    Float L0          = L0s(I);
                    Float h_bar       = h_bars(I);
                    Float torsion_bar = theta_torsion_bars(I);
                    Float bias_angle  = bias_angles(I);

                    Float yield_theta_warp   = yield_thetas_warp(I);
                    Float yield_theta_weft   = yield_thetas_weft(I);
                    Float plastic_theta_warp = plastic_thetas_warp(I);
                    Float plastic_theta_weft = plastic_thetas_weft(I);
                    Float anchor_theta_warp  = anchor_thetas_warp(I);
                    Float anchor_theta_weft  = anchor_thetas_weft(I);
                    Float thres_theta_warp   = thres_thetas_warp(I);
                    Float thres_theta_weft   = thres_thetas_weft(I);

                    Float V_bar = V_bars(I);

                    Vector3 x0 = xs(stencil[0]);
                    Vector3 x1 = xs(stencil[1]);
                    Vector3 x2 = xs(stencil[2]);
                    Vector3 x3 = xs(stencil[3]);

                    //printf("Vertex Index: %d, %d, %d, %d \r\n",
                    //       stencil[0],
                    //       stencil[1],
                    //       stencil[2],
                    //       stencil[3]);

                    Float Vdt2 = V_bar * dt * dt
                            * orthotropic_bending_normalization(bias_angle);

                    Vector12    G12;
                    Matrix12x12 H12x12;

                    DSGPB::dEdx(G12,
                                x0,
                                x1,
                                x2,
                                x3,
                                L0,
                                h_bar,
                                torsion_bar,
                                bias_angle,
                                plastic_theta_warp,
                                plastic_theta_weft,
                                anchor_theta_warp,
                                anchor_theta_weft,
                                kappa_warp,
                                kappa_weft,
                                bending_torsion,
                                bending_poisson,
                                friction_stiffness_warp,
                                friction_stiffness_weft);


                    G12 *= Vdt2;
                    assemble<4>(G3s, I * 4, stencil, G12);

                    DSGPB::ddEddx(H12x12,
                                  x0,
                                  x1,
                                  x2,
                                  x3,
                                  L0,
                                  h_bar,
                                  bias_angle,
                                  plastic_theta_warp,
                                  plastic_theta_weft,
                                  torsion_bar,
                                  anchor_theta_warp,
                                  anchor_theta_weft,
                                  kappa_warp,
                                  kappa_weft,
                                  bending_torsion,
                                  bending_poisson,
                                  friction_stiffness_warp,
                                  friction_stiffness_weft);

                    H12x12 *= Vdt2;





                    // if(!is_adjoint)



                    if(!audit_raw)
                        make_spd(H12x12);
                    assemble<4>(H3x3s, I * 4 * 4, stencil, H12x12);
                });
    }
};

REGISTER_SIM_SYSTEM(DiscreteShellFrictionHardeningPlasticOrthotropicBending);

class OrthotropicBendingHistoryDiffDofReporter final : public FiniteElementDiffDofReporter
{
  public:
    using FiniteElementDiffDofReporter::FiniteElementDiffDofReporter;
    DiscreteShellFrictionHardeningPlasticOrthotropicBending* constitution = nullptr;
    vector<Vector2i>             source_pairs;  // hinge, source position frame
    vector<Vector12>             source_gradients;
    vector<Float>                coefficients;
    muda::DeviceBuffer<Vector2i> pairs;
    muda::DeviceBuffer<Vector12> gradients;
    muda::DeviceBuffer<Float>    values;

    void do_build(BuildInfo&) override
    {
        constitution =
            &require<DiscreteShellFrictionHardeningPlasticOrthotropicBending>();
    }

    void do_report_extent(DiffDofExtentInfo& info) override
    {
        source_pairs.clear();
        source_gradients.clear();
        coefficients.clear();
        if(!constitution->history_position_gradient_enabled)
        {
            info.triplet_count(0);
            return;
        }
        SizeT F       = info.frame();
        auto& history = constitution->position_history;
        if(F <= 1 || constitution->stencils.size() == 0)
        {
            info.triplet_count(0);
            return;
        }
        UIPC_ASSERT(history.size() >= F, "Missing orthotropic bending transition history");


        for(SizeT I = 0; I < constitution->stencils.size(); ++I)
        {
            auto  C     = history[F - 1].force_coefficients[I];
            Float rp[2] = {C[0], C[1]}, rq[2] = {0.0, 0.0};
            Float ra[2] = {C[2], C[3]};
            for(SizeT update = F; update >= 2; --update)
            {
                auto&       snapshot    = history[update - 1];
                const auto& A           = snapshot.plastic_transition[I];
                const auto& B           = snapshot.anchor_transition[I];
                Float       coefficient = 0.0;
                for(int d = 0; d < 2; ++d)
                {
                    int o = 6 * d;
                    coefficient +=
                        rp[d] * A[o + 4] + rq[d] * A[o + 5] + ra[d] * B[2 * d + 1];
                    Float old_rp = rp[d], old_rq = rq[d];
                    rp[d] = old_rp * A[o] + old_rq * A[o + 2];
                    rq[d] = old_rp * A[o + 1] + old_rq * A[o + 3];
                    ra[d] *= B[2 * d];
                }
                if(coefficient != 0.0)
                {
                    source_pairs.push_back(Vector2i{IndexT(I), IndexT(update - 1)});
                    source_gradients.push_back(snapshot.source_gradient[I]);
                    coefficients.push_back(coefficient);
                }
            }
        }
        info.triplet_count(source_pairs.size() * 16);
    }

    void do_assemble(DiffDofInfo& info) override
    {
        using namespace muda;
        SizeT N = source_pairs.size();
        if(N == 0)
            return;
        vector<Vector2i> mapped = source_pairs;
        for(auto& pair : mapped)
            pair[1] = info.dof_offset(pair[1]) / 3;
        pairs.resize(N);
        gradients.resize(N);
        values.resize(N);
        pairs.view().copy_from(mapped.data());
        gradients.view().copy_from(source_gradients.data());
        values.view().copy_from(coefficients.data());
        using namespace sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;
        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(N,
                   [pairs = pairs.cviewer().name("bending_history_pairs"),
                    gs = gradients.cviewer().name("bending_history_source_gradient"),
                    coefficients = values.cviewer().name("bending_history_coefficient"),
                    idxs = constitution->stencils.cviewer().name("bending_history_stencils"),
                    xs = constitution->fem->xs().cviewer().name("bending_history_current_x"),
                    fixed = constitution->fem->is_fixed().cviewer().name("bending_history_fixed"),
                    H          = info.H().viewer().name("bending_history_H"),
                    row_offset = info.dof_offset(info.frame()) / 3,
                    dt         = info.dt()] __device__(int I) mutable
                   {
                       auto     pair = pairs(I);
                       auto     idx  = idxs(pair[0]);
                       Vector12 g;
                       dihedral_angle_gradient(
                           xs(idx[0]), xs(idx[1]), xs(idx[2]), xs(idx[3]), g);
                       Float coefficient = coefficients(I) * dt * dt;
                       for(int r = 0; r < 4; ++r)
                           for(int c = 0; c < 4; ++c)
                           {
                               Matrix3x3 block = Matrix3x3::Zero();
                               if(!fixed(idx[r]) && !fixed(idx[c]))
                                   block = coefficient * g.segment<3>(3 * r)
                                           * gs(I).segment<3>(3 * c).transpose();
                               H(I * 16 + r * 4 + c).write(row_offset + idx[r], pair[1] + idx[c], block);
                           }
                   });
    }
};

REGISTER_SIM_SYSTEM(OrthotropicBendingHistoryDiffDofReporter);


class DiffDiscreteShellFrictionHardeningPlasticOrthotropicBending final
    : public FiniteElementExtraConstitutionDiffParmReporter
{
  public:
    using Base = FiniteElementExtraConstitutionDiffParmReporter;
    using Base::Base;

    virtual U64 get_uid() const noexcept override
    {
        return DiscreteShellBendingUID;
    }

    virtual void do_build(BuildInfo& info) override {}

    DiscreteShellFrictionHardeningPlasticOrthotropicBending& constitution() noexcept
    {
        return static_cast<DiscreteShellFrictionHardeningPlasticOrthotropicBending&>(
            Base::constitution());
    }

    // Warp

    // Elastic
    vector<IndexT> h_diff_kappa_warp_parm_indices;
    vector<IndexT> h_diff_kappa_warp_stencil_indices;
    // Friction
    vector<IndexT> h_diff_fs_warp_parm_indices;
    vector<IndexT> h_diff_fs_warp_stencil_indices;
    vector<IndexT> h_diff_theta_t_warp_parm_indices;
    vector<IndexT> h_diff_theta_t_warp_stencil_indices;
    // Plastic
    vector<IndexT> h_diff_yield_ori_warp_parm_indices;
    vector<IndexT> h_diff_yield_ori_warp_stencil_indices;
    vector<IndexT> h_diff_hardening_k_warp_parm_indices;
    vector<IndexT> h_diff_hardening_k_warp_stencil_indices;

    // Weft
    // Elastic
    vector<IndexT> h_diff_kappa_weft_parm_indices;
    vector<IndexT> h_diff_kappa_weft_stencil_indices;
    // Friction
    vector<IndexT> h_diff_fs_weft_parm_indices;
    vector<IndexT> h_diff_fs_weft_stencil_indices;
    vector<IndexT> h_diff_theta_t_weft_parm_indices;
    vector<IndexT> h_diff_theta_t_weft_stencil_indices;
    // Plastic
    vector<IndexT> h_diff_yield_ori_weft_parm_indices;
    vector<IndexT> h_diff_yield_ori_weft_stencil_indices;
    vector<IndexT> h_diff_hardening_k_weft_parm_indices;
    vector<IndexT> h_diff_hardening_k_weft_stencil_indices;

    // Warp

    // Elastic (CUDA)
    muda::DeviceBuffer<IndexT> diff_kappa_warp_parm_indices;
    muda::DeviceBuffer<IndexT> diff_kappa_warp_stencil_indices;
    // Friction (CUDA)
    muda::DeviceBuffer<IndexT> diff_fs_warp_parm_indices;
    muda::DeviceBuffer<IndexT> diff_fs_warp_stencil_indices;
    muda::DeviceBuffer<IndexT> diff_theta_t_warp_parm_indices;
    muda::DeviceBuffer<IndexT> diff_theta_t_warp_stencil_indices;
    // Plastic (CUDA)
    muda::DeviceBuffer<IndexT> diff_yield_ori_warp_parm_indices;
    muda::DeviceBuffer<IndexT> diff_yield_ori_warp_stencil_indices;
    muda::DeviceBuffer<IndexT> diff_hardening_k_warp_parm_indices;
    muda::DeviceBuffer<IndexT> diff_hardening_k_warp_stencil_indices;

    // Weft

    // Elastic (CUDA)
    muda::DeviceBuffer<IndexT> diff_kappa_weft_parm_indices;
    muda::DeviceBuffer<IndexT> diff_kappa_weft_stencil_indices;
    // Friction (CUDA)
    muda::DeviceBuffer<IndexT> diff_fs_weft_parm_indices;
    muda::DeviceBuffer<IndexT> diff_fs_weft_stencil_indices;
    muda::DeviceBuffer<IndexT> diff_theta_t_weft_parm_indices;
    muda::DeviceBuffer<IndexT> diff_theta_t_weft_stencil_indices;
    // Plastic (CUDA)
    muda::DeviceBuffer<IndexT> diff_yield_ori_weft_parm_indices;
    muda::DeviceBuffer<IndexT> diff_yield_ori_weft_stencil_indices;
    muda::DeviceBuffer<IndexT> diff_hardening_k_weft_parm_indices;
    muda::DeviceBuffer<IndexT> diff_hardening_k_weft_stencil_indices;

    virtual void do_init(FilteredInfo& info) override
    {
        //std::cout << "Initialization in Backward Gated Plastic Bending" << std::endl;

        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        // kappa (warp)
        {
            list<IndexT> diff_kappa_warp_parm_indices_buffer;
            list<IndexT> diff_kappa_warp_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_kappa_warp =
                        sc.edges().find<IndexT>("diff/bending_stiffness_warp");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_kappa_warp)
                        return;

                    for(auto&& [diff_kappa_warp_v, E] : zip(diff_kappa_warp->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_kappa_warp_v >= 0 && init_info.valid_bending())
                        {
                            diff_kappa_warp_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                            diff_kappa_warp_parm_indices_buffer.push_back(diff_kappa_warp_v);
                        }
                    }
                });

            h_diff_kappa_warp_parm_indices.resize(
                diff_kappa_warp_parm_indices_buffer.size());
            h_diff_kappa_warp_stencil_indices.resize(
                diff_kappa_warp_stencil_indices_buffer.size());

            std::ranges::move(diff_kappa_warp_parm_indices_buffer,
                              h_diff_kappa_warp_parm_indices.begin());
            std::ranges::move(diff_kappa_warp_stencil_indices_buffer,
                              h_diff_kappa_warp_stencil_indices.begin());

            diff_kappa_warp_parm_indices.copy_from(h_diff_kappa_warp_parm_indices);
            diff_kappa_warp_stencil_indices.copy_from(h_diff_kappa_warp_stencil_indices);
        }

        std::cout << "Bending K (warp) Parameter Size: "
                  << h_diff_kappa_warp_parm_indices.size() << std::endl;

        // kappa (weft)
        {
            list<IndexT> diff_kappa_weft_parm_indices_buffer;
            list<IndexT> diff_kappa_weft_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_kappa_weft =
                        sc.edges().find<IndexT>("diff/bending_stiffness_weft");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_kappa_weft)
                        return;

                    for(auto&& [diff_kappa_weft_v, E] : zip(diff_kappa_weft->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_kappa_weft_v >= 0 && init_info.valid_bending())
                        {
                            diff_kappa_weft_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                            diff_kappa_weft_parm_indices_buffer.push_back(diff_kappa_weft_v);
                        }
                    }
                });

            h_diff_kappa_weft_parm_indices.resize(
                diff_kappa_weft_parm_indices_buffer.size());
            h_diff_kappa_weft_stencil_indices.resize(
                diff_kappa_weft_stencil_indices_buffer.size());

            std::ranges::move(diff_kappa_weft_parm_indices_buffer,
                              h_diff_kappa_weft_parm_indices.begin());
            std::ranges::move(diff_kappa_weft_stencil_indices_buffer,
                              h_diff_kappa_weft_stencil_indices.begin());

            diff_kappa_weft_parm_indices.copy_from(h_diff_kappa_weft_parm_indices);
            diff_kappa_weft_stencil_indices.copy_from(h_diff_kappa_weft_stencil_indices);
        }

        std::cout << "Bending K (weft) Parameter Size: "
                  << h_diff_kappa_weft_parm_indices.size() << std::endl;

        // friction stiffness (Warp)
        {
            list<IndexT> diff_fs_warp_parm_indices_buffer;
            list<IndexT> diff_fs_warp_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_fs_warp =
                        sc.edges().find<IndexT>("diff/friction_stiffness_warp");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_fs_warp)
                        return;

                    for(auto&& [diff_fs_warp_v, E] : zip(diff_fs_warp->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_fs_warp_v >= 0 && init_info.valid_bending())
                        {
                            diff_fs_warp_parm_indices_buffer.push_back(diff_fs_warp_v);
                            diff_fs_warp_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                        }
                    }
                });

            h_diff_fs_warp_parm_indices.resize(diff_fs_warp_parm_indices_buffer.size());
            h_diff_fs_warp_stencil_indices.resize(
                diff_fs_warp_stencil_indices_buffer.size());

            std::ranges::move(diff_fs_warp_parm_indices_buffer,
                              h_diff_fs_warp_parm_indices.begin());
            std::ranges::move(diff_fs_warp_stencil_indices_buffer,
                              h_diff_fs_warp_stencil_indices.begin());

            diff_fs_warp_parm_indices.resize(h_diff_fs_warp_parm_indices.size());
            diff_fs_warp_parm_indices.copy_from(h_diff_fs_warp_parm_indices);

            diff_fs_warp_stencil_indices.resize(h_diff_fs_warp_stencil_indices.size());
            diff_fs_warp_stencil_indices.copy_from(h_diff_fs_warp_stencil_indices);
        }

        std::cout << "Bending Friction Stiffness (warp) Size: "
                  << diff_fs_warp_parm_indices.size() << std::endl;

        // friction stiffness (Weff)
        {
            list<IndexT> diff_fs_weft_parm_indices_buffer;
            list<IndexT> diff_fs_weft_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_fs_weft =
                        sc.edges().find<IndexT>("diff/friction_stiffness_weft");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_fs_weft)
                        return;

                    for(auto&& [diff_fs_warp_v, E] : zip(diff_fs_weft->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_fs_warp_v >= 0 && init_info.valid_bending())
                        {
                            diff_fs_weft_parm_indices_buffer.push_back(diff_fs_warp_v);
                            diff_fs_weft_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                        }
                    }
                });

            h_diff_fs_weft_parm_indices.resize(diff_fs_weft_parm_indices_buffer.size());
            h_diff_fs_weft_stencil_indices.resize(
                diff_fs_weft_stencil_indices_buffer.size());

            std::ranges::move(diff_fs_weft_parm_indices_buffer,
                              h_diff_fs_weft_parm_indices.begin());
            std::ranges::move(diff_fs_weft_stencil_indices_buffer,
                              h_diff_fs_weft_stencil_indices.begin());

            diff_fs_weft_parm_indices.resize(h_diff_fs_weft_parm_indices.size());
            diff_fs_weft_parm_indices.copy_from(h_diff_fs_weft_parm_indices);

            diff_fs_weft_stencil_indices.resize(h_diff_fs_weft_stencil_indices.size());
            diff_fs_weft_stencil_indices.copy_from(h_diff_fs_weft_stencil_indices);
        }

        std::cout << "Bending Friction Stiffness (weft) Size: "
                  << diff_fs_weft_parm_indices.size() << std::endl;

        // Threshold Theta (warp)
        {
            list<IndexT> diff_theta_t_warp_parm_indices_buffer;
            list<IndexT> diff_theta_t_warp_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_theta_t_warp =
                        sc.edges().find<IndexT>("diff/thres_theta_warp");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_theta_t_warp)
                        return;

                    for(auto&& [diff_theta_t_warp_v, E] :
                        zip(diff_theta_t_warp->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_theta_t_warp_v >= 0 && init_info.valid_bending())
                        {
                            diff_theta_t_warp_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                            diff_theta_t_warp_parm_indices_buffer.push_back(diff_theta_t_warp_v);
                        }
                    }
                });

            h_diff_theta_t_warp_parm_indices.resize(
                diff_theta_t_warp_parm_indices_buffer.size());
            h_diff_theta_t_warp_stencil_indices.resize(
                diff_theta_t_warp_stencil_indices_buffer.size());

            std::ranges::move(diff_theta_t_warp_parm_indices_buffer,
                              h_diff_theta_t_warp_parm_indices.begin());
            std::ranges::move(diff_theta_t_warp_stencil_indices_buffer,
                              h_diff_theta_t_warp_stencil_indices.begin());

            diff_theta_t_warp_parm_indices.copy_from(h_diff_theta_t_warp_parm_indices);
            diff_theta_t_warp_stencil_indices.copy_from(h_diff_theta_t_warp_stencil_indices);
        }


        std::cout << "Bending Friction Thres Theta (warp) Size: "
                  << diff_theta_t_warp_parm_indices.size() << std::endl;

        // Threshold Theta (weft)
        {
            list<IndexT> diff_theta_t_weft_parm_indices_buffer;
            list<IndexT> diff_theta_t_weft_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_theta_t_weft =
                        sc.edges().find<IndexT>("diff/thres_theta_weft");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_theta_t_weft)
                        return;

                    for(auto&& [diff_theta_t_weft_v, E] :
                        zip(diff_theta_t_weft->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_theta_t_weft_v >= 0 && init_info.valid_bending())
                        {
                            diff_theta_t_weft_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                            diff_theta_t_weft_parm_indices_buffer.push_back(diff_theta_t_weft_v);
                        }
                    }
                });

            h_diff_theta_t_weft_parm_indices.resize(
                diff_theta_t_weft_parm_indices_buffer.size());
            h_diff_theta_t_weft_stencil_indices.resize(
                diff_theta_t_weft_stencil_indices_buffer.size());

            std::ranges::move(diff_theta_t_weft_parm_indices_buffer,
                              h_diff_theta_t_weft_parm_indices.begin());
            std::ranges::move(diff_theta_t_weft_stencil_indices_buffer,
                              h_diff_theta_t_weft_stencil_indices.begin());

            diff_theta_t_weft_parm_indices.copy_from(h_diff_theta_t_weft_parm_indices);
            diff_theta_t_weft_stencil_indices.copy_from(h_diff_theta_t_weft_stencil_indices);
        }

        std::cout << "Bending Friction Thres Theta (weft) Size: "
                  << diff_theta_t_weft_parm_indices.size() << std::endl;


        // Yield Ori (warp)
        {
            list<IndexT> diff_yield_ori_warp_parm_indices_buffer;
            list<IndexT> diff_yield_ori_warp_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_yield_theta_warp =
                        sc.edges().find<IndexT>("diff/yield_theta_ori_warp");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_yield_theta_warp)
                        return;

                    for(auto&& [diff_yield_theta_warp, E] :
                        zip(diff_yield_theta_warp->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_yield_theta_warp >= 0 && init_info.valid_bending())
                        {
                            diff_yield_ori_warp_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                            diff_yield_ori_warp_parm_indices_buffer.push_back(diff_yield_theta_warp);
                        }
                    }
                });

            h_diff_yield_ori_warp_parm_indices.resize(
                diff_yield_ori_warp_parm_indices_buffer.size());
            h_diff_yield_ori_warp_stencil_indices.resize(
                diff_yield_ori_warp_stencil_indices_buffer.size());

            std::ranges::move(diff_yield_ori_warp_parm_indices_buffer,
                              h_diff_yield_ori_warp_parm_indices.begin());
            std::ranges::move(diff_yield_ori_warp_stencil_indices_buffer,
                              h_diff_yield_ori_warp_stencil_indices.begin());

            diff_yield_ori_warp_parm_indices.copy_from(h_diff_yield_ori_warp_parm_indices);
            diff_yield_ori_warp_stencil_indices.copy_from(h_diff_yield_ori_warp_stencil_indices);
        }

        std::cout << "Bending Yield Theta Ori (warp) Size: "
                  << diff_yield_ori_warp_parm_indices.size() << std::endl;


        // Yield Ori (weft)
        {
            list<IndexT> diff_yield_ori_weft_parm_indices_buffer;
            list<IndexT> diff_yield_ori_weft_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_yield_theta_weft =
                        sc.edges().find<IndexT>("diff/yield_theta_ori_weft");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_yield_theta_weft)
                        return;

                    for(auto&& [diff_yield_theta_weft, E] :
                        zip(diff_yield_theta_weft->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_yield_theta_weft >= 0 && init_info.valid_bending())
                        {
                            diff_yield_ori_weft_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                            diff_yield_ori_weft_parm_indices_buffer.push_back(diff_yield_theta_weft);
                        }
                    }
                });

            h_diff_yield_ori_weft_parm_indices.resize(
                diff_yield_ori_weft_parm_indices_buffer.size());
            h_diff_yield_ori_weft_stencil_indices.resize(
                diff_yield_ori_weft_stencil_indices_buffer.size());

            std::ranges::move(diff_yield_ori_weft_parm_indices_buffer,
                              h_diff_yield_ori_weft_parm_indices.begin());
            std::ranges::move(diff_yield_ori_weft_stencil_indices_buffer,
                              h_diff_yield_ori_weft_stencil_indices.begin());

            diff_yield_ori_weft_parm_indices.copy_from(h_diff_yield_ori_weft_parm_indices);
            diff_yield_ori_weft_stencil_indices.copy_from(h_diff_yield_ori_weft_stencil_indices);
        }

        std::cout << "Bending Yield Theta Ori (weft) Size: "
                  << diff_yield_ori_weft_parm_indices.size() << std::endl;

        // Hardening k (warp)
        {
            list<IndexT> diff_hardening_k_warp_parm_indices_buffer;
            list<IndexT> diff_hardening_k_warp_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_hardening_k_warp =
                        sc.edges().find<IndexT>("diff/hardening_k_warp");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_hardening_k_warp)
                        return;

                    for(auto&& [diff_hardening_k_warp, E] :
                        zip(diff_hardening_k_warp->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_hardening_k_warp >= 0 && init_info.valid_bending())
                        {
                            diff_hardening_k_warp_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                            diff_hardening_k_warp_parm_indices_buffer.push_back(diff_hardening_k_warp);
                        }
                    }
                });

            h_diff_hardening_k_warp_parm_indices.resize(
                diff_hardening_k_warp_parm_indices_buffer.size());
            h_diff_hardening_k_warp_stencil_indices.resize(
                diff_hardening_k_warp_stencil_indices_buffer.size());

            std::ranges::move(diff_hardening_k_warp_parm_indices_buffer,
                              h_diff_hardening_k_warp_parm_indices.begin());
            std::ranges::move(diff_hardening_k_warp_stencil_indices_buffer,
                              h_diff_hardening_k_warp_stencil_indices.begin());

            diff_hardening_k_warp_parm_indices.copy_from(h_diff_hardening_k_warp_parm_indices);
            diff_hardening_k_warp_stencil_indices.copy_from(h_diff_hardening_k_warp_stencil_indices);
        }

        std::cout << "Bending Hardening k (warp) Size: "
                  << diff_hardening_k_warp_parm_indices.size() << std::endl;

        // Hardening k (weft)
        {
            list<IndexT> diff_hardening_k_weft_parm_indices_buffer;
            list<IndexT> diff_hardening_k_weft_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_hardening_k_weft =
                        sc.edges().find<IndexT>("diff/hardening_k_weft");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_hardening_k_weft)
                        return;

                    for(auto&& [diff_hardening_k_weft, E] :
                        zip(diff_hardening_k_weft->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_hardening_k_weft >= 0 && init_info.valid_bending())
                        {
                            diff_hardening_k_weft_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                            diff_hardening_k_weft_parm_indices_buffer.push_back(diff_hardening_k_weft);
                        }
                    }
                });

            h_diff_hardening_k_weft_parm_indices.resize(
                diff_hardening_k_weft_parm_indices_buffer.size());
            h_diff_hardening_k_weft_stencil_indices.resize(
                diff_hardening_k_weft_stencil_indices_buffer.size());

            std::ranges::move(diff_hardening_k_weft_parm_indices_buffer,
                              h_diff_hardening_k_weft_parm_indices.begin());
            std::ranges::move(diff_hardening_k_weft_stencil_indices_buffer,
                              h_diff_hardening_k_weft_stencil_indices.begin());

            diff_hardening_k_weft_parm_indices.copy_from(h_diff_hardening_k_weft_parm_indices);
            diff_hardening_k_weft_stencil_indices.copy_from(h_diff_hardening_k_weft_stencil_indices);
        }

        std::cout << "Bending Hardening k (weft) Size: "
                  << diff_hardening_k_weft_parm_indices.size() << std::endl;
    }

    virtual void do_update(DiffParmUpdateInfo& info) override
    {

        //std::cout << "In Do Update" << std::endl;

        using namespace muda;

        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_kappa_warp_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        kappa_warp_parm_indices =
                            diff_kappa_warp_parm_indices.viewer().name("kappa_warp_parm_indices"),
                        kappa_warp_stencil_indices =
                            diff_kappa_warp_stencil_indices.viewer().name("kappa_warp_stencil_indices"),
                        kappas_warp =
                            constitution().bending_stiffnesses_warp.viewer().name(
                                "bending_stiffness_warp")] __device__(int I)
                       {
                           auto& kappa_warp =
                               kappas_warp(kappa_warp_stencil_indices(I));
                           kappa_warp = parameters(kappa_warp_parm_indices(I));
                           //printf("kappa_warp updated value: %f", kappa_warp);
                       });
        }

        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_kappa_weft_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        kappa_weft_parm_indices =
                            diff_kappa_weft_parm_indices.viewer().name("kappa_weft_parm_indices"),
                        kappa_weft_stencil_indices =
                            diff_kappa_weft_stencil_indices.viewer().name("kappa_weft_stencil_indices"),
                        kappas_weft =
                            constitution().bending_stiffnesses_weft.viewer().name(
                                "bending_stiffness_weft")] __device__(int I)
                       {
                           auto& kappa_weft =
                               kappas_weft(kappa_weft_stencil_indices(I));
                           kappa_weft = parameters(kappa_weft_parm_indices(I));

                           //printf("kappa_weft updated value: %f", kappa_weft);
                       });
        }


        // Friction Stiffness (warp)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_fs_warp_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        fs_warp_parm_indices =
                            diff_fs_warp_parm_indices.viewer().name("fs_warp_parm_indices"),
                        fs_warp_stencil_indices =
                            diff_fs_warp_stencil_indices.viewer().name("fs_warp_stencil_indices"),
                        friciton_stiffnesses_warp =
                            constitution().friction_stiffnesses_warp.viewer().name(
                                "friction_stiffnesses_warp")] __device__(int I)
                       {
                           auto& friction_stiffness_warp =
                               friciton_stiffnesses_warp(fs_warp_stencil_indices(I));

                           //printf("friction_stiffness_warp BEFORE updated value: %f\r\n",
                           //       friction_stiffness_warp);

                           friction_stiffness_warp =
                               parameters(fs_warp_parm_indices(I));

                           //printf("friction_stiffness_warp Index: %d\r\n", fs_warp_parm_indices(I));

                           //printf("friction_stiffness_warp updated value: %f\r\n",
                           //       friction_stiffness_warp);
                       });
        }

        // Friction Stiffness (weft)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_fs_weft_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        fs_weft_parm_indices =
                            diff_fs_weft_parm_indices.viewer().name("fs_weft_parm_indices"),
                        fs_weft_stencil_indices =
                            diff_fs_weft_stencil_indices.viewer().name("fs_weft_stencil_indices"),
                        friciton_stiffnesses_weft =
                            constitution().friction_stiffnesses_weft.viewer().name(
                                "friction_stiffnesses_weft")] __device__(int I)
                       {
                           auto& friction_stiffness_weft =
                               friciton_stiffnesses_weft(fs_weft_stencil_indices(I));


                           //printf("friction_stiffness_weft BEFORE updated value: %f\r\n",
                           //       friction_stiffness_weft);

                           friction_stiffness_weft =
                               parameters(fs_weft_parm_indices(I));

                           //printf("friction_stiffness_weft Index: %d\r\n", fs_weft_parm_indices(I));


                           //printf("friction_stiffness_weft updated value: %f\r\n",
                           //       friction_stiffness_weft);
                       });
        }

        // Threshold theta (warp)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_theta_t_warp_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        theta_t_warp_parm_indices =
                            diff_theta_t_warp_parm_indices.viewer().name("theta_t_warp_parm_indices"),
                        theta_t_warp_stencil_indices =
                            diff_theta_t_warp_stencil_indices.viewer().name("theta_t_warp_stencil_indices"),
                        thres_thetas_warp =
                            constitution().thres_thetas_warp.viewer().name(
                                "thres_thetas_warp")] __device__(int I)
                       {
                           auto& thres_theta_warp =
                               thres_thetas_warp(theta_t_warp_stencil_indices(I));
                           thres_theta_warp = parameters(theta_t_warp_parm_indices(I));
                       });
        }

        // Threshold theta (weft)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_theta_t_weft_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        theta_t_weft_parm_indices =
                            diff_theta_t_weft_parm_indices.viewer().name("theta_t_weft_parm_indices"),
                        theta_t_weft_stencil_indices =
                            diff_theta_t_weft_stencil_indices.viewer().name("theta_t_weft_stencil_indices"),
                        thres_thetas_weft =
                            constitution().thres_thetas_weft.viewer().name(
                                "thres_thetas_weft")] __device__(int I)
                       {
                           auto& thres_theta_weft =
                               thres_thetas_weft(theta_t_weft_stencil_indices(I));
                           thres_theta_weft = parameters(theta_t_weft_parm_indices(I));
                       });
        }


        // Yield Ori (warp)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_yield_ori_warp_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        yield_ori_warp_parm_indices =
                            diff_yield_ori_warp_parm_indices.viewer().name("yield_ori_warp_parm_indices"),
                        yield_ori_warp_stencil_indices =
                            diff_yield_ori_warp_stencil_indices.viewer().name("yield_ori_warp_stencil_indices"),
                        yield_ori_thetas_warp =
                            constitution().yield_ori_thetas_warp.viewer().name(
                                "yield_theta_ori_warp")] __device__(int I)
                       {
                           auto& yield_theta_ori_warp =
                               yield_ori_thetas_warp(yield_ori_warp_stencil_indices(I));
                           yield_theta_ori_warp =
                               parameters(yield_ori_warp_parm_indices(I));
                       });
        }

        // Yield Ori (weft)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_yield_ori_weft_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        yield_ori_weft_parm_indices =
                            diff_yield_ori_weft_parm_indices.viewer().name("yield_ori_weft_parm_indices"),
                        yield_ori_weft_stencil_indices =
                            diff_yield_ori_weft_stencil_indices.viewer().name("yield_ori_weft_stencil_indices"),
                        yield_ori_thetas_weft =
                            constitution().yield_ori_thetas_weft.viewer().name(
                                "yield_theta_ori_weft")] __device__(int I)
                       {
                           auto& yield_theta_ori_weft =
                               yield_ori_thetas_weft(yield_ori_weft_stencil_indices(I));
                           yield_theta_ori_weft =
                               parameters(yield_ori_weft_parm_indices(I));
                       });
        }

        // Hardening k (warp)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_hardening_k_warp_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        hardening_k_warp_parm_indices =
                            diff_hardening_k_warp_parm_indices.viewer().name("hardening_k_warp_parm_indices"),
                        hardening_k_warp_stencil_indices =
                            diff_hardening_k_warp_stencil_indices.viewer().name("hardening_k_warp_stencil_indices"),
                        hardening_ks_warp =
                            constitution().hardening_ks_warp.viewer().name(
                                "hardening_k_warp")] __device__(int I)
                       {
                           auto& hardening_k_warp =
                               hardening_ks_warp(hardening_k_warp_stencil_indices(I));
                           hardening_k_warp =
                               parameters(hardening_k_warp_parm_indices(I));
                       });
        }

        // Hardening k (weft)
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_hardening_k_weft_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        hardening_k_weft_parm_indices =
                            diff_hardening_k_weft_parm_indices.viewer().name("hardening_k_weft_parm_indices"),
                        hardening_k_weft_stencil_indices =
                            diff_hardening_k_weft_stencil_indices.viewer().name("hardening_k_weft_stencil_indices"),
                        hardening_ks_weft =
                            constitution().hardening_ks_weft.viewer().name(
                                "hardening_k_weft")] __device__(int I)
                       {
                           auto& hardening_k_weft =
                               hardening_ks_weft(hardening_k_weft_stencil_indices(I));
                           hardening_k_weft =
                               parameters(hardening_k_weft_parm_indices(I));
                       });
        }
    }

    virtual void do_report_extent(DiffParmExtentInfo& info) override
    {
        //std::cout << "In Do Report extent" << std::endl;

        info.triplet_count(4 * diff_kappa_warp_stencil_indices.size()
                           + 4 * diff_fs_warp_stencil_indices.size()
                           + 4 * diff_theta_t_warp_stencil_indices.size()
                           + 4 * diff_yield_ori_warp_stencil_indices.size()
                           + 4 * diff_hardening_k_warp_stencil_indices.size()
                           + 4 * diff_kappa_weft_stencil_indices.size()
                           + 4 * diff_fs_weft_stencil_indices.size()
                           + 4 * diff_theta_t_weft_stencil_indices.size()
                           + 4 * diff_yield_ori_weft_stencil_indices.size()
                           + 4 * diff_hardening_k_weft_stencil_indices.size());
    }

    virtual void do_assemble(DiffParmInfo& info) override
    {
        using namespace muda;
        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_orthotropic_bending;

        auto pGpPs = info.pGpP();

        // assamble kappa (warp)
        {
            SizeT bending_k_warp_offset = 0;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_kappa_warp_stencil_indices.size(),
                    [diff_kappa_warp_stencil =
                         diff_kappa_warp_stencil_indices.viewer().name("diff_kappa_stencil_warp"),
                     diff_kappa_warp_parm_indices =
                         diff_kappa_warp_parm_indices.viewer().name("kappa_warp_parm_indices"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     bias_angles = constitution().bias_angles.cviewer().name("bias_angle"),
                     bending_stiffnesses_warp =
                         constitution().bending_stiffnesses_warp.cviewer().name("bending_stiffness_warp"),
                     plastic_thetas_warp = info.plastic_thetas_warp().viewer().name("plastic_theta_warp"),
                     d_plastic_theta_d_ks_warp =
                         info.d_plastic_theta_d_k_warp().viewer().name("d_plastic_theta_d_k_warp"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_k_warp_offset] __device__(int I) mutable
                    {
                        IndexT stencils_warp_index = diff_kappa_warp_stencil(I);
                        IndexT parm_warp_index = diff_kappa_warp_parm_indices(I);

                        Vector4i idx = stencils(stencils_warp_index);





                        // Float L0         = L0s(I);
                        // Float h_bar      = h_bars(I);
                        // Float bias_angle = bias_angles(I);
                        // Float bending_stiffness_warp = bending_stiffnesses_warp(I);
                        // Float plastic_theta_warp = plastic_thetas_warp(I);
                        // Float V_bar              = V_bars(I);

                        // Float d_plastic_theta_d_k_warp = d_plastic_theta_d_ks_warp(I);


                        Float L0         = L0s(stencils_warp_index);
                        Float h_bar      = h_bars(stencils_warp_index);
                        Float bias_angle = bias_angles(stencils_warp_index);
                        Float bending_stiffness_warp =
                            bending_stiffnesses_warp(stencils_warp_index);
                        Float plastic_theta_warp = plastic_thetas_warp(stencils_warp_index);
                        Float V_bar = V_bars(stencils_warp_index);

                        Float d_plastic_theta_d_k_warp =
                            d_plastic_theta_d_ks_warp(stencils_warp_index);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt
                            * orthotropic_bending_normalization(bias_angle);

                        Vector12 pGpKappaWarp;

                        DSGFPB::pGpKappaWarp(pGpKappaWarp,
                                             x0,
                                             x1,
                                             x2,
                                             x3,
                                             L0,
                                             h_bar,
                                             bias_angle,
                                             bending_stiffness_warp,
                                             plastic_theta_warp,
                                             d_plastic_theta_d_k_warp);

                        pGpKappaWarp *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpKappaWarp, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_warp_index;

                            pGpPs(bending_k_warp_offset + I * 4 + k)
                                .write(vI, J, pGpKappaWarp.segment<3>(k * 3));
                        }
                    });
        }


        // assamble kappa (weft)
        {
            SizeT bending_k_weft_offset = 4 * diff_kappa_warp_stencil_indices.size();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_kappa_weft_stencil_indices.size(),
                    [diff_kappa_weft_stencil =
                         diff_kappa_weft_stencil_indices.viewer().name("diff_kappa_stencil_weft"),
                     diff_kappa_weft_parm_indices =
                         diff_kappa_weft_parm_indices.viewer().name("kappa_weft_parm_indices"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     bias_angles = constitution().bias_angles.cviewer().name("bias_angle"),
                     bending_stiffnesses_weft =
                         constitution().bending_stiffnesses_weft.cviewer().name("bending_stiffness_weft"),
                     plastic_thetas_weft = info.plastic_thetas_weft().viewer().name("plastic_theta_weft"),
                     d_plastic_theta_d_ks_weft =
                         info.d_plastic_theta_d_k_weft().viewer().name("d_plastic_theta_d_k_weft"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_k_weft_offset] __device__(int I) mutable
                    {
                        IndexT stencils_weft_index = diff_kappa_weft_stencil(I);
                        IndexT parm_weft_index = diff_kappa_weft_parm_indices(I);

                        Vector4i idx = stencils(stencils_weft_index);





                        // Float L0         = L0s(I);
                        // Float h_bar      = h_bars(I);
                        // Float bias_angle = bias_angles(I);
                        // Float bending_stiffness_weft = bending_stiffnesses_weft(I);
                        // Float plastic_theta_weft = plastic_thetas_weft(I);
                        // Float V_bar              = V_bars(I);

                        // Float d_plastic_theta_d_k_weft = d_plastic_theta_d_ks_weft(I);


                        Float L0         = L0s(stencils_weft_index);
                        Float h_bar      = h_bars(stencils_weft_index);
                        Float bias_angle = bias_angles(stencils_weft_index);
                        Float bending_stiffness_weft =
                            bending_stiffnesses_weft(stencils_weft_index);
                        Float plastic_theta_weft = plastic_thetas_weft(stencils_weft_index);
                        Float V_bar = V_bars(stencils_weft_index);

                        Float d_plastic_theta_d_k_weft =
                            d_plastic_theta_d_ks_weft(stencils_weft_index);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt
                            * orthotropic_bending_normalization(bias_angle);

                        Vector12 pGpKappaWeft;

                        DSGFPB::pGpKappaWeft(pGpKappaWeft,
                                             x0,
                                             x1,
                                             x2,
                                             x3,
                                             L0,
                                             h_bar,
                                             bias_angle,
                                             bending_stiffness_weft,
                                             plastic_theta_weft,
                                             d_plastic_theta_d_k_weft);

                        pGpKappaWeft *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpKappaWeft, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_weft_index;

                            pGpPs(bending_k_weft_offset + I * 4 + k)
                                .write(vI, J, pGpKappaWeft.segment<3>(k * 3));
                        }
                    });
        }


        // Friction Stiffness (warp)
        {
            SizeT bending_friction_k_warp_offset =
                4 * diff_kappa_warp_stencil_indices.size()
                + 4 * diff_kappa_weft_stencil_indices.size();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_fs_warp_stencil_indices.size(),
                    [diff_fs_warp_stencil =
                         diff_fs_warp_stencil_indices.viewer().name("diff_fs_warp_stencil"),
                     diff_fs_warp_parm_indices =
                         diff_fs_warp_parm_indices.viewer().name("fsparm_warp_indices"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     bias_angles = constitution().bias_angles.cviewer().name("bias_angle"),
                     anchor_thetas_warp = info.anchor_thetas_warp().viewer().name("anchor_theta_warp"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_friction_k_warp_offset] __device__(int I) mutable
                    {
                        IndexT stencils_index_warp = diff_fs_warp_stencil(I);
                        IndexT parm_index_warp = diff_fs_warp_parm_indices(I);

                        Vector4i idx = stencils(stencils_index_warp);





                        // Float    L0         = L0s(I);
                        // Float    h_bar      = h_bars(I);
                        // Float    bias_angle = bias_angles(I);
                        // Float    anchor_theta_warp = anchor_thetas_warp(I);
                        // Float    V_bar             = V_bars(I);

                        Float L0         = L0s(stencils_index_warp);
                        Float h_bar      = h_bars(stencils_index_warp);
                        Float bias_angle = bias_angles(stencils_index_warp);
                        Float anchor_theta_warp = anchor_thetas_warp(stencils_index_warp);
                        Float V_bar = V_bars(stencils_index_warp);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt
                            * orthotropic_bending_normalization(bias_angle);

                        Vector12 pGpFrictionKWarp;
                        DSGFPB::pGpFrictionStiffnessWarp(
                            pGpFrictionKWarp, x0, x1, x2, x3, L0, h_bar, bias_angle, anchor_theta_warp);

                        pGpFrictionKWarp *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpFrictionKWarp, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_index_warp;

                            pGpPs(bending_friction_k_warp_offset + I * 4 + k)
                                .write(vI, J, pGpFrictionKWarp.segment<3>(k * 3));
                        }
                    });
        }


        // Friction Stiffness (weft)
        {
            SizeT bending_friction_k_weft_offset =
                4 * diff_kappa_warp_stencil_indices.size()
                + 4 * diff_kappa_weft_stencil_indices.size()
                + 4 * diff_fs_warp_stencil_indices.size();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_fs_weft_stencil_indices.size(),
                    [diff_fs_weft_stencil =
                         diff_fs_weft_stencil_indices.viewer().name("diff_fs_weft_stencil"),
                     diff_fs_weft_parm_indices =
                         diff_fs_weft_parm_indices.viewer().name("fsparm_weft_indices"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     bias_angles = constitution().bias_angles.cviewer().name("bias_angle"),
                     anchor_thetas_weft = info.anchor_thetas_weft().viewer().name("anchor_theta_weft"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_friction_k_weft_offset] __device__(int I) mutable
                    {
                        IndexT stencils_index_weft = diff_fs_weft_stencil(I);
                        IndexT parm_index_weft = diff_fs_weft_parm_indices(I);

                        Vector4i idx = stencils(stencils_index_weft);





                        // Float    L0         = L0s(I);
                        // Float    h_bar      = h_bars(I);
                        // Float    bias_angle = bias_angles(I);
                        // Float    anchor_theta_weft = anchor_thetas_weft(I);
                        // Float    V_bar             = V_bars(I);

                        Float L0         = L0s(stencils_index_weft);
                        Float h_bar      = h_bars(stencils_index_weft);
                        Float bias_angle = bias_angles(stencils_index_weft);
                        Float anchor_theta_weft = anchor_thetas_weft(stencils_index_weft);
                        Float V_bar = V_bars(stencils_index_weft);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt
                            * orthotropic_bending_normalization(bias_angle);

                        Vector12 pGpFrictionKWeft;
                        DSGFPB::pGpFrictionStiffnessWeft(
                            pGpFrictionKWeft, x0, x1, x2, x3, L0, h_bar, bias_angle, anchor_theta_weft);

                        pGpFrictionKWeft *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpFrictionKWeft, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_index_weft;

                            pGpPs(bending_friction_k_weft_offset + I * 4 + k)
                                .write(vI, J, pGpFrictionKWeft.segment<3>(k * 3));
                        }
                    });
        }

        // assumble theta_thres (warp)
        {
            SizeT bending_thres_warp_offset =
                4 * diff_kappa_warp_stencil_indices.size()
                + 4 * diff_kappa_weft_stencil_indices.size()
                + 4 * diff_fs_warp_stencil_indices.size()
                + 4 * diff_fs_weft_stencil_indices.size();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_theta_t_warp_stencil_indices.size(),
                    [diff_theta_t_stencil =
                         diff_theta_t_warp_stencil_indices.viewer().name("diff_theta_t_warp_stencil"),
                     diff_theta_t_parm_indices =
                         diff_theta_t_warp_parm_indices.viewer().name("theta_t_warp_parm_indices"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     bias_angles = constitution().bias_angles.cviewer().name("bias_angle"),
                     friction_stiffnesses_warp =
                         constitution().friction_stiffnesses_warp.viewer().name("friction_stiffness_warp"),
                     friction_stiffnesses_weft = constitution().friction_stiffnesses_weft.cviewer(),
                     cross_effective_threshold_sensitivity = info.d_anchor_thetas_d_thres_theta_weft().viewer().name("cross_effective_threshold_sensitivity"),
                     theta_bars = constitution().theta_bars.viewer().name("theta_bar"),
                     d_anchor_thetas_d_thres_thetas_warp =
                         info.d_anchor_thetas_d_thres_theta_warp().viewer().name("d_anchor_thetas_d_thres_theta_warp"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_thres_warp_offset] __device__(int I) mutable
                    {
                        IndexT stencils_index = diff_theta_t_stencil(I);
                        IndexT parm_index     = diff_theta_t_parm_indices(I);

                        Vector4i idx = stencils(stencils_index);





                        // Float friction_stiffness_warp = friction_stiffnesses_warp(I);
                        // Float d_anchor_theta_d_thres_theta_warp =


                        // Float L0         = L0s(I);
                        // Float h_bar      = h_bars(I);
                        // Float theta_bar  = theta_bars(I);
                        // Float V_bar      = V_bars(I);
                        // Float bias_angle = bias_angles(I);


                        Float friction_stiffness_warp =
                            friction_stiffnesses_warp(stencils_index);
                        Float d_anchor_theta_d_thres_theta_warp =
                            d_anchor_thetas_d_thres_thetas_warp(stencils_index);

                        Float L0         = L0s(stencils_index);
                        Float h_bar      = h_bars(stencils_index);
                        Float theta_bar  = theta_bars(stencils_index);
                        Float V_bar      = V_bars(stencils_index);
                        Float bias_angle = bias_angles(stencils_index);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt
                            * orthotropic_bending_normalization(bias_angle);

                        Vector12 pGpThetaYWarp;

                        Float c = cos(bias_angle + 1.570796);
                        Float sn = sin(bias_angle + 1.570796);
                        Float w = c * c, v = sn * sn;
                        Float dpw = (d_anchor_theta_d_thres_theta_warp) * (w * w);
                        Float dpv = (cross_effective_threshold_sensitivity(stencils_index)) * (w * v);
                        Float force_coefficient_w = friction_stiffnesses_warp(stencils_index) * w;
                        Float force_coefficient_v = friction_stiffnesses_weft(stencils_index) * v;
                        Vector12 angle_gradient;
                        uipc::backend::cuda::dihedral_angle_gradient(
                            x0, x1, x2, x3, angle_gradient);
                        pGpThetaYWarp = (-2.0 * L0 / h_bar)
                            * (force_coefficient_w * dpw + force_coefficient_v * dpv)
                            * angle_gradient;

                        pGpThetaYWarp *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpThetaYWarp, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_index;

                            pGpPs(bending_thres_warp_offset + I * 4 + k)
                                .write(vI, J, pGpThetaYWarp.segment<3>(k * 3));
                        }
                    });
        }

        // assumble theta_thres (weft)
        {
            SizeT bending_thres_weft_offset =
                4 * diff_kappa_warp_stencil_indices.size()
                + 4 * diff_kappa_weft_stencil_indices.size()
                + 4 * diff_fs_warp_stencil_indices.size()
                + 4 * diff_fs_weft_stencil_indices.size()
                + 4 * diff_theta_t_warp_stencil_indices.size();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_theta_t_weft_stencil_indices.size(),
                    [diff_theta_t_stencil =
                         diff_theta_t_weft_stencil_indices.viewer().name("diff_theta_t_weft_stencil"),
                     diff_theta_t_parm_indices =
                         diff_theta_t_weft_parm_indices.viewer().name("theta_t_weft_parm_indices"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     bias_angles = constitution().bias_angles.cviewer().name("bias_angle"),
                     friction_stiffnesses_weft =
                         constitution().friction_stiffnesses_weft.viewer().name("friction_stiffness_weft"),
                     friction_stiffnesses_warp = constitution().friction_stiffnesses_warp.cviewer(),
                     cross_effective_threshold_sensitivity = info.d_anchor_thetas_d_thres_theta_warp().viewer().name("cross_effective_threshold_sensitivity"),
                     theta_bars = constitution().theta_bars.viewer().name("theta_bar"),
                     d_anchor_thetas_d_thres_thetas_weft =
                         info.d_anchor_thetas_d_thres_theta_weft().viewer().name("d_anchor_thetas_d_thres_theta_weft"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_thres_weft_offset] __device__(int I) mutable
                    {
                        IndexT stencils_index = diff_theta_t_stencil(I);
                        IndexT parm_index     = diff_theta_t_parm_indices(I);

                        Vector4i idx = stencils(stencils_index);





                        // Float friction_stiffness_weft = friction_stiffnesses_weft(I);
                        // Float d_anchor_theta_d_thres_theta_weft =


                        // Float L0         = L0s(I);
                        // Float h_bar      = h_bars(I);
                        // Float theta_bar  = theta_bars(I);
                        // Float V_bar      = V_bars(I);
                        // Float bias_angle = bias_angles(I);


                        Float friction_stiffness_weft =
                            friction_stiffnesses_weft(stencils_index);
                        Float d_anchor_theta_d_thres_theta_weft =
                            d_anchor_thetas_d_thres_thetas_weft(stencils_index);

                        Float L0         = L0s(stencils_index);
                        Float h_bar      = h_bars(stencils_index);
                        Float theta_bar  = theta_bars(stencils_index);
                        Float V_bar      = V_bars(stencils_index);
                        Float bias_angle = bias_angles(stencils_index);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt
                            * orthotropic_bending_normalization(bias_angle);

                        Vector12 pGpThetaYWeft;

                        Float c = cos(bias_angle + 1.570796);
                        Float sn = sin(bias_angle + 1.570796);
                        Float w = c * c, v = sn * sn;
                        Float dpw = (cross_effective_threshold_sensitivity(stencils_index)) * (w * v);
                        Float dpv = (d_anchor_theta_d_thres_theta_weft) * (v * v);
                        Float force_coefficient_w = friction_stiffnesses_warp(stencils_index) * w;
                        Float force_coefficient_v = friction_stiffnesses_weft(stencils_index) * v;
                        Vector12 angle_gradient;
                        uipc::backend::cuda::dihedral_angle_gradient(
                            x0, x1, x2, x3, angle_gradient);
                        pGpThetaYWeft = (-2.0 * L0 / h_bar)
                            * (force_coefficient_w * dpw + force_coefficient_v * dpv)
                            * angle_gradient;

                        pGpThetaYWeft *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpThetaYWeft, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_index;

                            pGpPs(bending_thres_weft_offset + I * 4 + k)
                                .write(vI, J, pGpThetaYWeft.segment<3>(k * 3));
                        }
                    });
        }


        // assamble yield_ori (warp)
        {
            SizeT bending_yield_ori_warp_offset =
                4 * diff_kappa_warp_stencil_indices.size()
                + 4 * diff_kappa_weft_stencil_indices.size()
                + 4 * diff_fs_warp_stencil_indices.size()
                + 4 * diff_fs_weft_stencil_indices.size()
                + 4 * diff_theta_t_warp_stencil_indices.size()
                + 4 * diff_theta_t_weft_stencil_indices.size();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_yield_ori_warp_stencil_indices.size(),
                    [diff_yield_ori_warp_stencil =
                         diff_yield_ori_warp_stencil_indices.viewer().name("diff_yield_ori_warp_stencil"),
                     diff_yield_ori_warp_parm_indices =
                         diff_yield_ori_warp_parm_indices.viewer().name("yield_ori_warp_parm_indices"),
                     bending_stiffnesses_warp =
                         constitution().bending_stiffnesses_warp.viewer().name("bending_stiffness_warp"),
                     bending_stiffnesses_weft = constitution().bending_stiffnesses_weft.cviewer(),
                     bending_poissons = constitution().bending_poissons.cviewer(),
                     cross_effective_yield_sensitivity = info.d_plastic_theta_d_yield_ori_weft().viewer().name("cross_effective_yield_sensitivity"),
                     d_plastic_theta_d_yield_oris_warp =
                         info.d_plastic_theta_d_yield_ori_warp().viewer().name("d_plastic_theta_d_yield_ori_warp"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     bias_angles = constitution().bias_angles.cviewer().name("bias_angle"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_yield_ori_warp_offset] __device__(int I) mutable
                    {
                        IndexT stencils_index_warp = diff_yield_ori_warp_stencil(I);
                        IndexT parm_index_warp = diff_yield_ori_warp_parm_indices(I);

                        Vector4i idx = stencils(stencils_index_warp);





                        // Float L0         = L0s(I);
                        // Float h_bar      = h_bars(I);
                        // Float V_bar      = V_bars(I);
                        // Float bias_angle = bias_angles(I);

                        // Float bending_stiffness_warp = bending_stiffnesses_warp(I);
                        // Float d_plastic_theta_d_yield_ori_warp =



                        Float L0         = L0s(stencils_index_warp);
                        Float h_bar      = h_bars(stencils_index_warp);
                        Float V_bar      = V_bars(stencils_index_warp);
                        Float bias_angle = bias_angles(stencils_index_warp);

                        Float bending_stiffness_warp =
                            bending_stiffnesses_warp(stencils_index_warp);
                        Float d_plastic_theta_d_yield_ori_warp =
                            d_plastic_theta_d_yield_oris_warp(stencils_index_warp);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt
                            * orthotropic_bending_normalization(bias_angle);

                        Vector12 pGpYieldOriWarp;

                        Float c = cos(bias_angle + 1.570796);
                        Float sn = sin(bias_angle + 1.570796);
                        Float w = c * c, v = sn * sn;
                        Float dpw = (d_plastic_theta_d_yield_ori_warp) * (w * w);
                        Float dpv = (cross_effective_yield_sensitivity(stencils_index_warp)) * (w * v);
                        Float force_coefficient_w = bending_stiffnesses_warp(stencils_index_warp) * w + bending_poissons(stencils_index_warp) * v;
                        Float force_coefficient_v = bending_stiffnesses_weft(stencils_index_warp) * v + bending_poissons(stencils_index_warp) * w;
                        Vector12 angle_gradient;
                        uipc::backend::cuda::dihedral_angle_gradient(
                            x0, x1, x2, x3, angle_gradient);
                        pGpYieldOriWarp = (-2.0 * L0 / h_bar)
                            * (force_coefficient_w * dpw + force_coefficient_v * dpv)
                            * angle_gradient;

                        //printf("Backward d_plastic_theta_d_yield_ori_warp: %f\n\r",
                        //       d_plastic_theta_d_yield_ori_warp);

                        //printf("pGpYieldOriWarp: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\r\n",
                        //       pGpYieldOriWarp(0),
                        //       pGpYieldOriWarp(1),
                        //       pGpYieldOriWarp(2),
                        //       pGpYieldOriWarp(3),
                        //       pGpYieldOriWarp(4),
                        //       pGpYieldOriWarp(5),
                        //       pGpYieldOriWarp(6),
                        //       pGpYieldOriWarp(7),
                        //       pGpYieldOriWarp(8),
                        //       pGpYieldOriWarp(9),
                        //       pGpYieldOriWarp(10),
                        //       pGpYieldOriWarp(11));

                        pGpYieldOriWarp *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpYieldOriWarp, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_index_warp;

                            pGpPs(bending_yield_ori_warp_offset + I * 4 + k)
                                .write(vI, J, pGpYieldOriWarp.segment<3>(k * 3));
                        }
                    });
        }


        // assamble yield_ori (weft)
        {
            SizeT bending_yield_ori_weft_offset =
                4 * diff_kappa_warp_stencil_indices.size()
                + 4 * diff_kappa_weft_stencil_indices.size()
                + 4 * diff_fs_warp_stencil_indices.size()
                + 4 * diff_fs_weft_stencil_indices.size()
                + 4 * diff_theta_t_warp_stencil_indices.size()
                + 4 * diff_theta_t_weft_stencil_indices.size()
                + 4 * diff_yield_ori_warp_stencil_indices.size();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_yield_ori_weft_stencil_indices.size(),
                    [diff_yield_ori_weft_stencil =
                         diff_yield_ori_weft_stencil_indices.viewer().name("diff_yield_ori_weft_stencil"),
                     diff_yield_ori_weft_parm_indices =
                         diff_yield_ori_weft_parm_indices.viewer().name("yield_ori_weft_parm_indices"),
                     bending_stiffnesses_weft =
                         constitution().bending_stiffnesses_weft.viewer().name("bending_stiffness_weft"),
                     bending_stiffnesses_warp = constitution().bending_stiffnesses_warp.cviewer(),
                     bending_poissons = constitution().bending_poissons.cviewer(),
                     cross_effective_yield_sensitivity = info.d_plastic_theta_d_yield_ori_warp().viewer().name("cross_effective_yield_sensitivity"),
                     d_plastic_theta_d_yield_oris_weft =
                         info.d_plastic_theta_d_yield_ori_weft().viewer().name("d_plastic_theta_d_yield_ori_weft"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     bias_angles = constitution().bias_angles.cviewer().name("bias_angle"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_yield_ori_weft_offset] __device__(int I) mutable
                    {
                        IndexT stencils_index_weft = diff_yield_ori_weft_stencil(I);
                        IndexT parm_index_weft = diff_yield_ori_weft_parm_indices(I);

                        Vector4i idx = stencils(stencils_index_weft);





                        // Float L0         = L0s(I);
                        // Float h_bar      = h_bars(I);
                        // Float V_bar      = V_bars(I);
                        // Float bias_angle = bias_angles(I);

                        // Float bending_stiffness_weft = bending_stiffnesses_weft(I);
                        // Float d_plastic_theta_d_yield_ori_weft =






                        Float L0         = L0s(stencils_index_weft);
                        Float h_bar      = h_bars(stencils_index_weft);
                        Float V_bar      = V_bars(stencils_index_weft);
                        Float bias_angle = bias_angles(stencils_index_weft);

                        Float bending_stiffness_weft =
                            bending_stiffnesses_weft(stencils_index_weft);
                        Float d_plastic_theta_d_yield_ori_weft =
                            d_plastic_theta_d_yield_oris_weft(stencils_index_weft);

                        //printf("Backward d_plastic_theta_d_yield_ori_weft: %f\n\r",
                        //       d_plastic_theta_d_yield_ori_weft);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt
                            * orthotropic_bending_normalization(bias_angle);

                        Vector12 pGpYieldOriWeft;

                        Float c = cos(bias_angle + 1.570796);
                        Float sn = sin(bias_angle + 1.570796);
                        Float w = c * c, v = sn * sn;
                        Float dpw = (cross_effective_yield_sensitivity(stencils_index_weft)) * (w * v);
                        Float dpv = (d_plastic_theta_d_yield_ori_weft) * (v * v);
                        Float force_coefficient_w = bending_stiffnesses_warp(stencils_index_weft) * w + bending_poissons(stencils_index_weft) * v;
                        Float force_coefficient_v = bending_stiffnesses_weft(stencils_index_weft) * v + bending_poissons(stencils_index_weft) * w;
                        Vector12 angle_gradient;
                        uipc::backend::cuda::dihedral_angle_gradient(
                            x0, x1, x2, x3, angle_gradient);
                        pGpYieldOriWeft = (-2.0 * L0 / h_bar)
                            * (force_coefficient_w * dpw + force_coefficient_v * dpv)
                            * angle_gradient;

                        //printf("pGpYieldOriWeft: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\r\n",
                        //       pGpYieldOriWeft(0),
                        //       pGpYieldOriWeft(1),
                        //       pGpYieldOriWeft(2),
                        //       pGpYieldOriWeft(3),
                        //       pGpYieldOriWeft(4),
                        //       pGpYieldOriWeft(5),
                        //       pGpYieldOriWeft(6),
                        //       pGpYieldOriWeft(7),
                        //       pGpYieldOriWeft(8),
                        //       pGpYieldOriWeft(9),
                        //       pGpYieldOriWeft(10),
                        //       pGpYieldOriWeft(11));

                        pGpYieldOriWeft *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpYieldOriWeft, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_index_weft;

                            pGpPs(bending_yield_ori_weft_offset + I * 4 + k)
                                .write(vI, J, pGpYieldOriWeft.segment<3>(k * 3));
                        }
                    });
        }

        // assamble hardening_k (warp)
        {
            SizeT bending_hardening_k_warp_offset =
                4 * diff_kappa_warp_stencil_indices.size()
                + 4 * diff_kappa_weft_stencil_indices.size()
                + 4 * diff_fs_warp_stencil_indices.size()
                + 4 * diff_fs_weft_stencil_indices.size()
                + 4 * diff_theta_t_warp_stencil_indices.size()
                + 4 * diff_theta_t_weft_stencil_indices.size()
                + 4 * diff_yield_ori_warp_stencil_indices.size()
                + 4 * diff_yield_ori_weft_stencil_indices.size();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_hardening_k_warp_stencil_indices.size(),
                    [diff_hardening_k_warp_stencil =
                         diff_hardening_k_warp_stencil_indices.viewer().name("diff_hardening_k_stencil_warp"),
                     diff_hardening_k_warp_parm_indices =
                         diff_hardening_k_warp_parm_indices.viewer().name("hardening_k_warp_parm_indices"),
                     bending_stiffnesses_warp =
                         constitution().bending_stiffnesses_warp.viewer().name("bending_stiffness_warp"),
                     d_plastic_theta_d_hardening_ks_warp =
                         info.d_plastic_theta_d_hardening_k_warp().viewer().name("d_plastic_theta_d_hardening_k_warp"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     bias_angles = constitution().bias_angles.cviewer().name("bias_angle"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_hardening_k_warp_offset] __device__(int I) mutable
                    {
                        IndexT stencils_index_warp = diff_hardening_k_warp_stencil(I);
                        IndexT parm_index_warp = diff_hardening_k_warp_parm_indices(I);

                        Vector4i idx = stencils(stencils_index_warp);





                        // Float L0         = L0s(I);
                        // Float h_bar      = h_bars(I);
                        // Float V_bar      = V_bars(I);
                        // Float bias_angle = bias_angles(I);

                        // Float bending_stiffness_warp = bending_stiffnesses_warp(I);
                        // Float d_plastic_theta_d_hardening_k_warp =



                        Float L0         = L0s(stencils_index_warp);
                        Float h_bar      = h_bars(stencils_index_warp);
                        Float V_bar      = V_bars(stencils_index_warp);
                        Float bias_angle = bias_angles(stencils_index_warp);

                        Float bending_stiffness_warp =
                            bending_stiffnesses_warp(stencils_index_warp);
                        Float d_plastic_theta_d_hardening_k_warp =
                            d_plastic_theta_d_hardening_ks_warp(stencils_index_warp);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt
                            * orthotropic_bending_normalization(bias_angle);

                        Vector12 pGpHardeningKWarp;
                        DSGFPB::pGpHardeningKWarp(
                            pGpHardeningKWarp, x0, x1, x2, x3, L0, h_bar, bias_angle, bending_stiffness_warp, d_plastic_theta_d_hardening_k_warp);

                        pGpHardeningKWarp *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpHardeningKWarp, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_index_warp;

                            pGpPs(bending_hardening_k_warp_offset + I * 4 + k)
                                .write(vI, J, pGpHardeningKWarp.segment<3>(k * 3));
                        }
                    });
        }

        // assamble hardening_k (weft)
        {
            SizeT bending_hardening_k_weft_offset =
                4 * diff_kappa_warp_stencil_indices.size()
                + 4 * diff_kappa_weft_stencil_indices.size()
                + 4 * diff_fs_warp_stencil_indices.size()
                + 4 * diff_fs_weft_stencil_indices.size()
                + 4 * diff_theta_t_warp_stencil_indices.size()
                + 4 * diff_theta_t_weft_stencil_indices.size()
                + 4 * diff_yield_ori_warp_stencil_indices.size()
                + 4 * diff_yield_ori_weft_stencil_indices.size()
                + 4 * diff_hardening_k_warp_stencil_indices.size();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_hardening_k_weft_stencil_indices.size(),
                    [diff_hardening_k_weft_stencil =
                         diff_hardening_k_weft_stencil_indices.viewer().name("diff_hardening_k_stencil_weft"),
                     diff_hardening_k_weft_parm_indices =
                         diff_hardening_k_weft_parm_indices.viewer().name("hardening_k_weft_parm_indices"),
                     bending_stiffnesses_weft =
                         constitution().bending_stiffnesses_weft.viewer().name("bending_stiffness_weft"),
                     d_plastic_theta_d_hardening_ks_weft =
                         info.d_plastic_theta_d_hardening_k_weft().viewer().name("d_plastic_theta_d_hardening_k_weft"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     bias_angles = constitution().bias_angles.cviewer().name("bias_angle"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_hardening_k_weft_offset] __device__(int I) mutable
                    {
                        IndexT stencils_index_weft = diff_hardening_k_weft_stencil(I);
                        IndexT parm_index_weft = diff_hardening_k_weft_parm_indices(I);

                        Vector4i idx = stencils(stencils_index_weft);





                        // Float L0         = L0s(I);
                        // Float h_bar      = h_bars(I);
                        // Float V_bar      = V_bars(I);
                        // Float bias_angle = bias_angles(I);

                        // Float bending_stiffness_weft = bending_stiffnesses_weft(I);
                        // Float d_plastic_theta_d_hardening_k_weft =



                        Float L0         = L0s(stencils_index_weft);
                        Float h_bar      = h_bars(stencils_index_weft);
                        Float V_bar      = V_bars(stencils_index_weft);
                        Float bias_angle = bias_angles(stencils_index_weft);

                        Float bending_stiffness_weft =
                            bending_stiffnesses_weft(stencils_index_weft);
                        Float d_plastic_theta_d_hardening_k_weft =
                            d_plastic_theta_d_hardening_ks_weft(stencils_index_weft);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt
                            * orthotropic_bending_normalization(bias_angle);

                        Vector12 pGpHardeningKWeft;
                        DSGFPB::pGpHardeningKWeft(
                            pGpHardeningKWeft, x0, x1, x2, x3, L0, h_bar, bias_angle, bending_stiffness_weft, d_plastic_theta_d_hardening_k_weft);

                        pGpHardeningKWeft *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpHardeningKWeft, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_index_weft;

                            pGpPs(bending_hardening_k_weft_offset + I * 4 + k)
                                .write(vI, J, pGpHardeningKWeft.segment<3>(k * 3));
                        }
                    });
        }
    }
};

REGISTER_SIM_SYSTEM(DiffDiscreteShellFrictionHardeningPlasticOrthotropicBending);

}  // namespace uipc::backend::cuda
