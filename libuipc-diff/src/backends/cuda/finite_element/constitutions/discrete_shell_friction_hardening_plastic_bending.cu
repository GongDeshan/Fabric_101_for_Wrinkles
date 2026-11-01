#include <finite_element/finite_element_extra_constitution.h>
#include <uipc/builtin/attribute_name.h>
#include <finite_element/constitutions/discrete_shell_friction_hardening_plastic_bending_function.h>
#include <numbers>
#include <utils/matrix_assembly_utils.h>
#include <kernel_cout.h>
#include <finite_element/finite_element_extra_constitution_diff_parm_reporter.h>

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
static constexpr U64 DiscreteShellBendingUID = 23;

class DiscreteShellFrictionHardeningPlasticBending final : public FiniteElementExtraConstitution
{
    using Base = FiniteElementExtraConstitution;
    friend class DiscreteShellFrictionHardeningPlasticBending;

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
    vector<Float>    h_bending_stiffness;
    vector<Float>    h_rest_volumes;
    vector<Float>    h_rest_lengths;
    vector<Float>    h_h_bars;
    vector<Float>    h_theta_bars;
    vector<Float>    h_V_bars;

    // For (perfect) plastic parameters
    vector<Float> h_hardening_ks;
    vector<Float> h_hardening_plastic_thetas;
    vector<Float> h_dihedral_thetas;
    vector<Float> h_plastic_thetas;
    vector<Float> h_yield_thetas;

    // For (Columb) friction parameters
    vector<Float> h_friction_stiffness;
    vector<Float> h_thres_thetas;
    vector<Float> h_anchor_thetas;

    muda::DeviceBuffer<Vector4i> stencils;  // X0, X1, X2, X3; (X1, X2) is middle edge
    muda::DeviceBuffer<Float>         bending_stiffnesses;
    muda::DeviceBuffer<Float>         rest_lengths;
    muda::DeviceBuffer<Float>         h_bars;
    muda::DeviceBuffer<Float>         theta_bars;
    muda::DeviceBuffer<Float>         V_bars;
    unordered_map<Vector2i, InitInfo> stencil_map;  // Edge -> opposite vertices

    // Plastic Parameter in GPU
    muda::DeviceBuffer<Float> hardening_ks;
    muda::DeviceBuffer<Float> hardening_plastic_thetes;
    muda::DeviceBuffer<Float> dihedral_thetas;
    muda::DeviceBuffer<Float> plastic_thetas;
    muda::DeviceBuffer<Float> yield_ori_thetas, yield_thetas;



    muda::DeviceBuffer<Vector3> plastic_parameter_sensitivities;
    muda::DeviceBuffer<Vector3> accumulated_plastic_parameter_sensitivities;

    // Friction Parameter in GPU
    muda::DeviceBuffer<Float> friction_stiffnesses;
    muda::DeviceBuffer<Float> thres_thetas;
    muda::DeviceBuffer<Float> anchor_thetas;

    // Gates of the Weight
    muda::DeviceBuffer<Float> weights;

    FiniteElementMethod*                 fem;
    std::span<S<geometry::GeometrySlot>> geo_slots;
    vector<bool>                         is_valid_bending;

    vector<IndexT> filter_geo_slot_indices;

    // MyFeature

    void friction_plastic_update()
    {
        //std::cout << "Friction Hardening Plastic Bending friction plastic update starts"
        //          << std::endl;

        using namespace muda;
        using namespace sym::discrete_shell_friction_hardening_plastic_bending;

        // Used when single object
        auto geo = geo_slots.front()->geometry().as<geometry::SimplicialComplex>();

        // Used when sim multiple objects
        //auto geo = geo_slots[filter_geo_slot_indices[0]]->geometry().as<geometry::SimplicialComplex>();

        auto rest_anchor_plastic =
            geo->edges().find<I32>("reset_anchor_plastic")->view();

        if(rest_anchor_plastic[0] != 0)
        {
            auto plastic_thetas_view =
                geo->edges().find<Float>("plastic_theta")->view();
            auto anchor_thetas_view = geo->edges().find<Float>("anchor_theta")->view();
            //auto yield_theta_view = geo->edges().find<Float>("yield_theta")->view();

            int idx_cnt{0};

            int cnt{0};

            for(int i = 0; i < is_valid_bending.size(); ++i)
            {
                if(is_valid_bending[i])
                {
                    h_anchor_thetas[idx_cnt] = anchor_thetas_view[cnt];

                    //h_yield_thetas[idx_cnt]   = yield_theta_view[cnt];
                    h_plastic_thetas[idx_cnt] = plastic_thetas_view[cnt];

                    idx_cnt++;
                }
                cnt++;
            }

            anchor_thetas.view().copy_from(h_anchor_thetas.data());

            //yield_thetas.view().copy_from(h_yield_thetas.data());
            plastic_thetas.view().copy_from(h_plastic_thetas.data());
            hardening_plastic_thetes.fill(0.0);
            plastic_parameter_sensitivities.fill(Vector3{0.0, 0.0, 0.0});
            accumulated_plastic_parameter_sensitivities.fill(Vector3{0.0, 0.0, 0.0});

            yield_thetas = yield_ori_thetas;

            fem->d_plastic_theta_d_yield_ori.fill(0.0);
            fem->d_plastic_theta_d_k.fill(0.0);
            fem->d_plastic_theta_d_hardening_k.fill(0.0);
            fem->d_anchor_theta_d_thres_theta.fill(0.0);
        }

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(
                stencils.size(),
                [stencils = stencils.viewer().name("stencils"),
                 bending_stiffnesses = bending_stiffnesses.viewer().name("bending_stiffness"),
                 dihedral_thetas = dihedral_thetas.viewer().name("dihedral_theta"),
                 anchor_thetas = anchor_thetas.viewer().name("anchor_theta"),
                 hardening_ks  = hardening_ks.viewer().name("hardening_k"),
                 yield_thetas  = yield_thetas.viewer().name("yield_theta"),
                 yield_ori_thetas = yield_ori_thetas.viewer().name("yield_ori_theta"),
                 hardening_plastic_thetas = hardening_plastic_thetes.viewer().name("hardening_plastic_theta"),
                 plastic_thetas = plastic_thetas.viewer().name("plastic_theta"),
                 dp_parameters = plastic_parameter_sensitivities.viewer().name("dp_parameters"),
                 dq_parameters = accumulated_plastic_parameter_sensitivities.viewer().name("dq_parameters"),
                 thres_thetas   = thres_thetas.viewer().name("thres_theta"),
                 fem_plastic_thetas = fem->plastic_thetas.viewer().name("fem_plastic_theta"),
                 fem_anchor_thetas = fem->anchor_thetas.viewer().name("fem_anchor_theta"),
                 fem_dihedral_thetas = fem->dihedral_thetas.viewer().name("fem_dihedral_theta"),
                 fem_d_anchor_theta_d_thres_theta =
                     fem->d_anchor_theta_d_thres_theta.viewer().name("fem_d_anchor_theta_d_thres_theta"),
                 fem_d_plastic_theta_d_yield_ori =
                     fem->d_plastic_theta_d_yield_ori.viewer().name("fem_d_plastic_theta_d_yield_ori"),
                 fem_d_plastic_theta_d_k = fem->d_plastic_theta_d_k.viewer().name("fem_d_plastic_theta_d_k"),
                 fem_d_plastic_theta_d_hardening_k =
                     fem->d_plastic_theta_d_hardening_k.viewer().name("fem_d_plastic_theta_d_hardening_k"),
                 xs = fem->xs().viewer().name("xs")] __device__(int I)
                {
                    Vector4i stencil = stencils(I);

                    Float k  = bending_stiffnesses(I);
                    Float hk = hardening_ks(I);

                    // Friction Params
                    Float anchor_theta = anchor_thetas(I);
                    Float thres_theta  = thres_thetas(I);

                    // Plastic Params
                    Float plastic_theta           = plastic_thetas(I);
                    Float hardening_plastic_theta = hardening_plastic_thetas(I);
                    Float yield_ori_theta         = yield_ori_thetas(I);
                    //Float yield_theta             = yield_thetas(I);

                    Vector3 x0 = xs(stencil[0]);
                    Vector3 x1 = xs(stencil[1]);
                    Vector3 x2 = xs(stencil[2]);
                    Vector3 x3 = xs(stencil[3]);
                    Float   theta;
                    dihedral_angle(x0, x1, x2, x3, theta);

                    dihedral_thetas(I) = theta;

                    Float yield_theta =
                        yield_ori_theta + hardening_plastic_theta * (hk / k);

                    Float elastic_theta = theta - plastic_theta;
                    bool  is_yield      = abs(elastic_theta) > yield_theta;

                    if(is_yield)
                    {
                        plastic_thetas(I) = plastic_theta
                                            + sign(elastic_theta)
                                                  * (abs(elastic_theta) - yield_theta)
                                                  * (k / (k + hk));

                        hardening_plastic_thetas(I) =
                            hardening_plastic_theta
                            + (abs(elastic_theta) - yield_theta) * (k / (k + hk));

                        yield_thetas(I) =
                            yield_ori_theta + hardening_plastic_thetas(I) * (hk / k);










                        const Float s = sign(elastic_theta);
                        const Float r = k / (k + hk);
                        const Float excess = abs(elastic_theta) - yield_theta;
                        const Float denom_sq = (k + hk) * (k + hk);
                        const Vector3 dp_old = dp_parameters(I);
                        const Vector3 dq_old = dq_parameters(I);
                        Vector3 d_yield;
                        d_yield(0) = 1.0 + (hk / k) * dq_old(0);
                        d_yield(1) = (hk / k) * dq_old(1)
                                     - hardening_plastic_theta * hk / (k * k);
                        d_yield(2) = (hk / k) * dq_old(2)
                                     + hardening_plastic_theta / k;
                        Vector3 d_ratio;
                        d_ratio(0) = 0.0;
                        d_ratio(1) = hk / denom_sq;
                        d_ratio(2) = -k / denom_sq;
                        Vector3 dp_new, dq_new;
                        for(int j = 0; j < 3; ++j)
                        {
                            const Float d_gamma =
                                r * (-s * dp_old(j) - d_yield(j))
                                + excess * d_ratio(j);
                            dp_new(j) = dp_old(j) + s * d_gamma;
                            dq_new(j) = dq_old(j) + d_gamma;
                        }
                        dp_parameters(I) = dp_new;
                        dq_parameters(I) = dq_new;
                    }



                    fem_d_plastic_theta_d_yield_ori(I) = dp_parameters(I)(0);
                    fem_d_plastic_theta_d_k(I) = dp_parameters(I)(1);
                    fem_d_plastic_theta_d_hardening_k(I) = dp_parameters(I)(2);

                    Float var_theta = theta - anchor_theta;
                    bool  is_slide  = abs(var_theta) > thres_theta;
                    if(is_slide)
                    {
                        anchor_thetas(I) =
                            anchor_theta + sign(var_theta) * (abs(var_theta) - thres_theta);

                        fem_d_anchor_theta_d_thres_theta(I) = -sign(var_theta);
                    }



                    fem_dihedral_thetas(I) = theta;
                    fem_plastic_thetas(I)  = plastic_thetas(I);
                    fem_anchor_thetas(I)   = anchor_thetas(I);
                });

        dihedral_thetas.view().copy_to(h_dihedral_thetas.data());
        anchor_thetas.view().copy_to(h_anchor_thetas.data());
        plastic_thetas.view().copy_to(h_plastic_thetas.data());
        hardening_plastic_thetes.view().copy_to(h_hardening_plastic_thetas.data());
        yield_thetas.view().copy_to(h_yield_thetas.data());

        auto dihedral_theta_view = view(*geo->edges().find<Float>("dihedral_theta"));
        auto plastic_theta_view = view(*geo->edges().find<Float>("plastic_theta"));
        auto anchor_theta_view = view(*geo->edges().find<Float>("anchor_theta"));
        auto yield_theta_view = view(*geo->edges().find<Float>("yield_theta"));
        auto hardening_plastic_theta_view =
            view(*geo->edges().find<Float>("hardening_plastic_theta"));

        auto dihedral_theta_view_it = dihedral_theta_view.begin();
        auto plastic_theta_view_it  = plastic_theta_view.begin();
        auto anchor_theta_view_it   = anchor_theta_view.begin();
        auto yield_theta_view_it    = yield_theta_view.begin();
        auto hardening_plastic_theta_view_it = hardening_plastic_theta_view.begin();

        int cnt{0};

        for(int i = 0; i < is_valid_bending.size(); ++i)
        {
            if(is_valid_bending.at(i))
            {
                *dihedral_theta_view_it = h_dihedral_thetas[cnt];
                *plastic_theta_view_it  = h_plastic_thetas[cnt];
                *anchor_theta_view_it   = h_anchor_thetas[cnt];
                *yield_theta_view_it    = h_yield_thetas[cnt];
                *hardening_plastic_theta_view_it = h_hardening_plastic_thetas[cnt];
                cnt++;
            }

            dihedral_theta_view_it++;
            plastic_theta_view_it++;
            anchor_theta_view_it++;
            yield_theta_view_it++;
            hardening_plastic_theta_view_it++;
        }

        //std::cout << "Friction Hardening Plastic Bending friction plastic update Done"
        //          << std::endl;
    }

    virtual void do_build(BuildInfo& info) override
    {
        fem = &require<FiniteElementMethod>();

        auto& dof_predictor = require<DofPredictor>();

        dof_predictor.on_predict(*this,
                                 [&](DofPredictor::PredictInfo& info)
                                 { friction_plastic_update(); });  // Update Plastic Theta
    }

    virtual void do_init(FilteredInfo& info) override
    {
        std::cout << "Friction Hardening Plastic Bending Model Init" << std::endl;

        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_bending;

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
        list<Float>    bending_stiffness_list;
        list<Float>    hardening_k_list;

        list<Float> dihedral_theta_list;
        list<Float> plastic_theta_list;
        list<Float> yield_theta_list;

        list<Float> friction_stiffness_list;
        list<Float> thres_theta_list;
        list<Float> anchor_theta_list;

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

                auto bending_stiffnesses = sc.edges().find<Float>("bending_stiffness");
                UIPC_ASSERT(bending_stiffnesses, "Bending stiffness not found, why?");
                auto bs_view = bending_stiffnesses->view();

                auto hardening_ks = sc.edges().find<Float>("hardening_k");
                UIPC_ASSERT(hardening_ks, "Hardening_k not found, why?");
                auto hks_view = hardening_ks->view();

                auto dihedral_theta = sc.edges().find<Float>("dihedral_theta");
                UIPC_ASSERT(dihedral_theta, "Dihedral Theta not found, why?");
                auto ds_view = dihedral_theta->view();

                // Plastic Properties
                auto plastic_thetas = sc.edges().find<Float>("plastic_theta");
                UIPC_ASSERT(plastic_thetas, "Plastic theta is not found!");
                auto pt_view = plastic_thetas->view();

                auto hardening_plastic_theta_attr =
                    sc.edges().find<Float>("hardening_plastic_theta");
                if(!hardening_plastic_theta_attr)
                    hardening_plastic_theta_attr =
                        sc.edges().create<Float>("hardening_plastic_theta", 0.0);
                std::ranges::fill(geometry::view(*hardening_plastic_theta_attr), 0.0);

                auto yield_thetas = sc.edges().find<Float>("yield_theta");
                UIPC_ASSERT(yield_thetas, "Yield theta is not found!");
                auto yt_view = yield_thetas->view();

                // Find Friction Properties
                auto friction_stiffnesses = sc.edges().find<Float>("friction_stiffness");
                UIPC_ASSERT(friction_stiffnesses, "Friction Stiffness not found, why?");
                auto fs_view = friction_stiffnesses->view();

                auto thres_thetas = sc.edges().find<Float>("thres_theta");
                UIPC_ASSERT(thres_thetas, "Threshold Theta not found, why?");
                auto tt_view = thres_thetas->view();

                auto anchor_thetas = sc.edges().find<Float>("anchor_theta");
                UIPC_ASSERT(anchor_thetas, "Anchor Theta not found, why?");
                auto at_view = anchor_thetas->view();

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

                auto stencils = sc.edges().find<Vector4i>("stencil");
                if(!stencils)
                    stencils = sc.edges().create<Vector4i>("stencil");
                auto stencils_view = geometry::view(*stencils);
                std::ranges::fill(stencils_view, Vector4i::Zero());

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

                        Float bs = bs_view[info.edge_index];
                        bending_stiffness_list.push_back(bs);

                        Float hk = hks_view[info.edge_index];
                        hardening_k_list.push_back(hk);

                        Float ds = ds_view[info.edge_index];
                        dihedral_theta_list.push_back(ds);

                        Float pt = pt_view[info.edge_index];
                        plastic_theta_list.push_back(pt);

                        Float yt = yt_view[info.edge_index];
                        yield_theta_list.push_back(yt);

                        // Friction
                        Float fs = fs_view[info.edge_index];
                        friction_stiffness_list.push_back(fs);

                        Float tt = tt_view[info.edge_index];
                        thres_theta_list.push_back(tt);

                        Float acht = at_view[info.edge_index];
                        anchor_theta_list.push_back(acht);

                        is_valid_bending.push_back(true);
                    }
                    else
                    {
                        is_valid_bending.push_back(false);
                    }
                }
            });

        // 2) Setup Invariant Data
        h_stencils.resize(stencil_list.size());
        h_bending_stiffness.resize(stencil_list.size());
        std::ranges::move(stencil_list, h_stencils.begin());
        std::ranges::move(bending_stiffness_list, h_bending_stiffness.begin());
        //      2.1): Plastic Parameters
        h_hardening_ks.resize(stencil_list.size());
        h_yield_thetas.resize(stencil_list.size());
        h_plastic_thetas.resize(stencil_list.size());
        h_dihedral_thetas.resize(stencil_list.size());
        std::ranges::move(hardening_k_list, h_hardening_ks.begin());
        std::ranges::move(yield_theta_list, h_yield_thetas.begin());
        std::ranges::move(plastic_theta_list, h_plastic_thetas.begin());
        std::ranges::move(dihedral_theta_list, h_dihedral_thetas.begin());
        //      2.2): Friction Parameters
        h_friction_stiffness.resize(stencil_list.size());
        h_anchor_thetas.resize(stencil_list.size());
        h_thres_thetas.resize(stencil_list.size());
        std::ranges::move(friction_stiffness_list, h_friction_stiffness.begin());
        std::ranges::move(anchor_theta_list, h_anchor_thetas.begin());
        std::ranges::move(thres_theta_list, h_thres_thetas.begin());

        // 3) Setup Related Data
        span x_bars      = info.rest_positions();
        span thicknesses = info.thicknesses();
        h_rest_lengths.resize(h_stencils.size());
        h_h_bars.resize(h_stencils.size());
        h_theta_bars.resize(h_stencils.size());
        h_V_bars.resize(h_stencils.size());

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

            h_rest_lengths[i] = L0;
            h_h_bars[i]       = h_bar;
            h_theta_bars[i]   = 0.0;
            h_V_bars[i]       = V_bar;
        }


        //Used when simulate single object
        //Initialize the theta angles.
        auto dihedral_theta_view = view(*geo->edges().find<Float>("dihedral_theta"));
        auto plastic_theta_view = view(*geo->edges().find<Float>("plastic_theta"));
        auto anchor_theta_view = view(*geo->edges().find<Float>("anchor_theta"));

        auto dihedral_theta_view_it = dihedral_theta_view.begin();
        auto plastic_theta_view_it  = plastic_theta_view.begin();
        auto anchor_theta_view_it   = anchor_theta_view.begin();

        auto L0s_view      = view(*geo->edges().find<Float>("L0"));
        auto V_bars_view   = view(*geo->edges().find<Float>("V_bar"));
        auto h_bars_view   = view(*geo->edges().find<Float>("h_bar"));
        auto stencils_view = view(*geo->edges().find<Vector4i>("stencil"));

        auto L0s_view_it      = L0s_view.begin();
        auto V_bars_view_it   = V_bars_view.begin();
        auto h_bars_view_it   = h_bars_view.begin();
        auto stencils_view_it = stencils_view.begin();

        int cnt{0};

        for(int i = 0; i < is_valid_bending.size(); ++i)
        {
            if(is_valid_bending.at(i))
            {
                *dihedral_theta_view_it = h_dihedral_thetas[cnt];
                *plastic_theta_view_it  = h_dihedral_thetas[cnt];
                *anchor_theta_view_it   = h_dihedral_thetas[cnt];

                *L0s_view_it      = h_rest_lengths[cnt];
                *V_bars_view_it   = h_V_bars[cnt];
                *h_bars_view_it   = h_h_bars[cnt];
                *stencils_view_it = h_stencils[cnt];
                cnt++;
            }

            dihedral_theta_view_it++;
            plastic_theta_view_it++;
            anchor_theta_view_it++;

            L0s_view_it++;
            V_bars_view_it++;
            h_bars_view_it++;
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

        bending_stiffnesses.resize(h_bending_stiffness.size());
        bending_stiffnesses.view().copy_from(h_bending_stiffness.data());

        hardening_ks.resize(h_hardening_ks.size());
        hardening_ks.view().copy_from(h_hardening_ks.data());

        dihedral_thetas.resize(h_dihedral_thetas.size());
        dihedral_thetas.view().copy_from(h_dihedral_thetas.data());

        fem->dihedral_thetas.resize(dihedral_thetas.size());
        fem->dihedral_thetas.view().copy_from(h_dihedral_thetas.data());

        // Plastic Properties
        hardening_plastic_thetes.resize(h_hardening_ks.size());
        hardening_plastic_thetes.fill(0.0);
        plastic_parameter_sensitivities.resize(h_hardening_ks.size());
        plastic_parameter_sensitivities.fill(Vector3{0.0, 0.0, 0.0});
        accumulated_plastic_parameter_sensitivities.resize(h_hardening_ks.size());
        accumulated_plastic_parameter_sensitivities.fill(Vector3{0.0, 0.0, 0.0});
        h_hardening_plastic_thetas.resize(h_hardening_ks.size());
        hardening_plastic_thetes.view().copy_to(h_hardening_plastic_thetas.data());

        yield_thetas.resize(h_yield_thetas.size());
        yield_thetas.view().copy_from(h_yield_thetas.data());

        yield_ori_thetas.resize(h_yield_thetas.size());
        yield_ori_thetas.view().copy_from(h_yield_thetas.data());

        plastic_thetas.resize(h_plastic_thetas.size());
        plastic_thetas.view().copy_from(h_dihedral_thetas.data());
        // Plastic: Save for backward
        fem->plastic_thetas.resize(h_plastic_thetas.size());
        fem->plastic_thetas.view().copy_from(h_dihedral_thetas.data());

        fem->d_plastic_theta_d_yield_ori.resize(h_plastic_thetas.size());
        fem->d_plastic_theta_d_yield_ori.fill(0.0);

        fem->d_plastic_theta_d_k.resize(h_plastic_thetas.size());
        fem->d_plastic_theta_d_k.fill(0.0);

        fem->d_plastic_theta_d_hardening_k.resize(h_plastic_thetas.size());
        fem->d_plastic_theta_d_hardening_k.fill(0.0);

        // Friction Properties
        friction_stiffnesses.resize(h_friction_stiffness.size());
        friction_stiffnesses.view().copy_from(h_friction_stiffness.data());

        thres_thetas.resize(h_thres_thetas.size());
        thres_thetas.view().copy_from(h_thres_thetas.data());

        anchor_thetas.resize(h_anchor_thetas.size());
        anchor_thetas.view().copy_from(h_dihedral_thetas.data());

        fem->anchor_thetas.resize(h_anchor_thetas.size());
        fem->anchor_thetas.view().copy_from(h_dihedral_thetas.data());

        fem->d_anchor_theta_d_thres_theta.resize(h_anchor_thetas.size());  // Save gradient for backward
        fem->d_anchor_theta_d_thres_theta.fill(0.0);  // Save gradient for backward

        rest_lengths.resize(h_rest_lengths.size());
        rest_lengths.view().copy_from(h_rest_lengths.data());

        h_bars.resize(h_h_bars.size());
        h_bars.view().copy_from(h_h_bars.data());

        theta_bars.resize(h_theta_bars.size());
        theta_bars.view().copy_from(h_theta_bars.data());

        V_bars.resize(h_V_bars.size());
        V_bars.view().copy_from(h_V_bars.data());
    }

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        info.energy_count(stencils.size());  // Each quad has 1 energy
        info.stencil_dim(4);                 // Each quad has 4 vertices
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        using namespace muda;
        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_bending;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(info.energies().size(),
                   [stencils = stencils.viewer().name("stencils"),
                    bending_stiffnesses = bending_stiffnesses.viewer().name("bending_stiffness"),
                    theta_bars = theta_bars.viewer().name("theta_bar"),
                    plastic_thetas = plastic_thetas.viewer().name("plastic_theta"),
                    yield_thetas = yield_thetas.viewer().name("yield_theta"),
                    friction_stiffnesses = friction_stiffnesses.viewer().name("friction_stiffness"),
                    anchor_thetas = anchor_thetas.viewer().name("anchor_theta"),
                    thres_thetas  = thres_thetas.viewer().name("thres_theta"),
                    h_bars        = h_bars.viewer().name("h_bar"),
                    V_bars        = V_bars.viewer().name("V_bar"),
                    L0s           = rest_lengths.viewer().name("rest_lengths"),
                    xs            = info.xs().viewer().name("xs"),
                    energies      = info.energies().viewer().name("energies"),
                    weights       = weights.viewer().name("weights"),
                    dt            = info.dt()] __device__(int I)
                   {
                       Vector4i stencil            = stencils(I);
                       Float    kappa              = bending_stiffnesses(I);
                       Float    L0                 = L0s(I);
                       Float    h_bar              = h_bars(I);
                       Float    theta_bar          = theta_bars(I);
                       Float    plastic_theta      = plastic_thetas(I);
                       Float    friction_stiffness = friction_stiffnesses(I);
                       Float    yield_theta        = yield_thetas(I);
                       Float    anchor_theta       = anchor_thetas(I);
                       Float    thres_theta        = thres_thetas(I);
                       Float    V_bar              = V_bars(I);

                       Vector3 x0 = xs(stencil[0]);
                       Vector3 x1 = xs(stencil[1]);
                       Vector3 x2 = xs(stencil[2]);
                       Vector3 x3 = xs(stencil[3]);

                       // Using weight to predict/calculate the: plastic_theta
                       Float E = DSGFPB::E(x0,
                                           x1,
                                           x2,
                                           x3,
                                           L0,
                                           h_bar,
                                           theta_bar,
                                           yield_theta,
                                           plastic_theta,
                                           anchor_theta,
                                           thres_theta,
                                           kappa,
                                           friction_stiffness);

                       energies(I) = E * V_bar * dt * dt;
                   });
    }

    virtual void do_compute_gradient_hessian(ComputeGradientHessianInfo& info) override
    {
        using namespace muda;
        namespace DSGPB = sym::discrete_shell_friction_hardening_plastic_bending;




        auto& plastic_theta_state =
            info.is_adjoint() ? fem->plastic_thetas : plastic_thetas;
        auto& anchor_theta_state =
            info.is_adjoint() ? fem->anchor_thetas : anchor_thetas;

        ParallelFor()
            .file_line(__FILE__, __LINE__)
            .apply(stencils.size(),
                   [stencils = stencils.viewer().name("stencils"),
                    bending_stiffnesses = bending_stiffnesses.viewer().name("bending_stiffness"),
                    theta_bars   = theta_bars.viewer().name("theta_bar"),
                    yield_thetas = yield_thetas.viewer().name("yield_theta"),
                    // Old: plastic_thetas = plastic_thetas.viewer().name("plastic_theta"),
                    plastic_thetas = plastic_theta_state.viewer().name("plastic_theta"),
                    friction_stiffnesses = friction_stiffnesses.viewer().name("friction_stiffness"),
                    // Old: anchor_thetas = anchor_thetas.viewer().name("anchor_theta"),
                    anchor_thetas = anchor_theta_state.viewer().name("anchor_theta"),
                    thres_thetas  = thres_thetas.viewer().name("thres_theta"),
                    thicknesses = info.thicknesses().viewer().name("thicknesses"),
                    h_bars = h_bars.viewer().name("h_bar"),
                    V_bars = V_bars.viewer().name("V_bar"),
                    L0s    = rest_lengths.viewer().name("rest_lengths"),
                    xs     = info.xs().viewer().name("xs"),
                    G3s    = info.gradients().viewer().name("gradients"),
                    H3x3s  = info.hessians().viewer().name("hessians"),
                    dt     = info.dt(),
                    is_adjoint = info.is_adjoint()] __device__(int I) mutable
                   {
                       Vector4i stencil            = stencils(I);
                       Float    kappa              = bending_stiffnesses(I);
                       Float    friction_stiffness = friction_stiffnesses(I);
                       Float    L0                 = L0s(I);
                       Float    h_bar              = h_bars(I);
                       Float    theta_bar          = theta_bars(I);
                       Float    yield_theta        = yield_thetas(I);
                       Float    plastic_theta      = plastic_thetas(I);
                       Float    anchor_theta       = anchor_thetas(I);
                       Float    thres_theta        = thres_thetas(I);
                       Float    V_bar              = V_bars(I);

                       Vector3 x0 = xs(stencil[0]);
                       Vector3 x1 = xs(stencil[1]);
                       Vector3 x2 = xs(stencil[2]);
                       Vector3 x3 = xs(stencil[3]);

                       Float Vdt2 = V_bar * dt * dt;

                       Vector12    G12;
                       Matrix12x12 H12x12;

                       DSGPB::dEdx(G12,
                                   x0,
                                   x1,
                                   x2,
                                   x3,
                                   L0,
                                   h_bar,
                                   theta_bar,
                                   yield_theta,
                                   plastic_theta,
                                   anchor_theta,
                                   thres_theta,
                                   kappa,
                                   friction_stiffness);
                       G12 *= Vdt2;
                       assemble<4>(G3s, I * 4, stencil, G12);

                       DSGPB::ddEddx(H12x12,
                                     x0,
                                     x1,
                                     x2,
                                     x3,
                                     L0,
                                     h_bar,
                                     theta_bar,
                                     yield_theta,
                                     plastic_theta,
                                     anchor_theta,
                                     thres_theta,
                                     kappa,
                                     friction_stiffness);

                       H12x12 *= Vdt2;





                       // if(!is_adjoint)

                       make_spd(H12x12);
                       assemble<4>(H3x3s, I * 4 * 4, stencil, H12x12);
                   });
    }
};

REGISTER_SIM_SYSTEM(DiscreteShellFrictionHardeningPlasticBending);

class DiffDiscreteShellFrictionHardeningPlasticBending final : public FiniteElementExtraConstitutionDiffParmReporter
{
  public:
    using Base = FiniteElementExtraConstitutionDiffParmReporter;
    using Base::Base;

    virtual U64 get_uid() const noexcept override
    {
        return DiscreteShellBendingUID;
    }

    virtual void do_build(BuildInfo& info) override {}

    DiscreteShellFrictionHardeningPlasticBending& constitution() noexcept
    {
        return static_cast<DiscreteShellFrictionHardeningPlasticBending&>(Base::constitution());
    }

    vector<IndexT> h_diff_kappa_parm_indices;
    vector<IndexT> h_diff_kappa_stencil_indices;
    // Friction
    vector<IndexT> h_diff_fs_parm_indices;
    vector<IndexT> h_diff_fs_stencil_indices;
    vector<IndexT> h_diff_theta_t_parm_indices;
    vector<IndexT> h_diff_theta_t_stencil_indices;
    // Plastic
    vector<IndexT> h_diff_yield_ori_parm_indices;
    vector<IndexT> h_diff_yield_ori_stencil_indices;
    vector<IndexT> h_diff_hardening_k_parm_indices;
    vector<IndexT> h_diff_hardening_k_stencil_indices;

    muda::DeviceBuffer<IndexT> diff_kappa_parm_indices;
    muda::DeviceBuffer<IndexT> diff_kappa_stencil_indices;
    // Friction (CUDA)
    muda::DeviceBuffer<IndexT> diff_fs_parm_indices;
    muda::DeviceBuffer<IndexT> diff_fs_stencil_indices;
    muda::DeviceBuffer<IndexT> diff_theta_t_parm_indices;
    muda::DeviceBuffer<IndexT> diff_theta_t_stencil_indices;
    // Plastic (CUDA)
    muda::DeviceBuffer<IndexT> diff_yield_ori_parm_indices;
    muda::DeviceBuffer<IndexT> diff_yield_ori_stencil_indices;
    muda::DeviceBuffer<IndexT> diff_hardening_k_parm_indices;
    muda::DeviceBuffer<IndexT> diff_hardening_k_stencil_indices;

    virtual void do_init(FilteredInfo& info) override
    {
        //std::cout << "Initialization in Backward Gated Plastic Bending" << std::endl;

        using ForEachInfo = FiniteElementMethod::ForEachInfo;

        auto geo_slots = world().scene().geometries();

        // kappa
        {
            list<IndexT> diff_kappa_parm_indices_buffer;
            list<IndexT> diff_kappa_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_kappa = sc.edges().find<IndexT>("diff/bending_stiffness");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_kappa)
                        return;

                    for(auto&& [diff_kappa_v, E] : zip(diff_kappa->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_kappa_v >= 0 && init_info.valid_bending())
                        {
                            diff_kappa_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                            diff_kappa_parm_indices_buffer.push_back(diff_kappa_v);
                        }
                    }
                });

            h_diff_kappa_parm_indices.resize(diff_kappa_parm_indices_buffer.size());
            h_diff_kappa_stencil_indices.resize(
                diff_kappa_stencil_indices_buffer.size());

            std::ranges::move(diff_kappa_parm_indices_buffer,
                              h_diff_kappa_parm_indices.begin());
            std::ranges::move(diff_kappa_stencil_indices_buffer,
                              h_diff_kappa_stencil_indices.begin());

            diff_kappa_parm_indices.copy_from(h_diff_kappa_parm_indices);
            diff_kappa_stencil_indices.copy_from(h_diff_kappa_stencil_indices);
        }

        std::cout << "Bending K Parameter Size: " << diff_kappa_parm_indices.size()
                  << std::endl;

        // friction stiffness
        {
            list<IndexT> diff_fs_parm_indices_buffer;
            list<IndexT> diff_fs_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_fs = sc.edges().find<IndexT>("diff/friction_stiffness");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_fs)
                        return;

                    for(auto&& [diff_fs_v, E] : zip(diff_fs->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_fs_v >= 0 && init_info.valid_bending())
                        {
                            diff_fs_stencil_indices_buffer.push_back(init_info.stencil_index);
                            diff_fs_parm_indices_buffer.push_back(diff_fs_v);
                        }
                    }
                });

            h_diff_fs_parm_indices.resize(diff_fs_parm_indices_buffer.size());
            h_diff_fs_stencil_indices.resize(diff_fs_stencil_indices_buffer.size());

            std::ranges::move(diff_fs_parm_indices_buffer,
                              h_diff_fs_parm_indices.begin());
            std::ranges::move(diff_fs_stencil_indices_buffer,
                              h_diff_fs_stencil_indices.begin());

            diff_fs_parm_indices.copy_from(h_diff_fs_parm_indices);
            diff_fs_stencil_indices.copy_from(h_diff_fs_stencil_indices);
        }

        std::cout << "Bending Friction Stiffness Size: " << diff_fs_parm_indices.size()
                  << std::endl;


        // Threshold Theta
        {
            list<IndexT> diff_theta_t_parm_indices_buffer;
            list<IndexT> diff_theta_t_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_theta_t = sc.edges().find<IndexT>("diff/thres_theta");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_theta_t)
                        return;

                    for(auto&& [diff_theta_t_v, E] : zip(diff_theta_t->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_theta_t_v >= 0 && init_info.valid_bending())
                        {
                            diff_theta_t_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                            diff_theta_t_parm_indices_buffer.push_back(diff_theta_t_v);
                        }
                    }
                });

            h_diff_theta_t_parm_indices.resize(diff_theta_t_parm_indices_buffer.size());
            h_diff_theta_t_stencil_indices.resize(
                diff_theta_t_stencil_indices_buffer.size());

            std::ranges::move(diff_theta_t_parm_indices_buffer,
                              h_diff_theta_t_parm_indices.begin());
            std::ranges::move(diff_theta_t_stencil_indices_buffer,
                              h_diff_theta_t_stencil_indices.begin());

            diff_theta_t_parm_indices.copy_from(h_diff_theta_t_parm_indices);
            diff_theta_t_stencil_indices.copy_from(h_diff_theta_t_stencil_indices);
        }


        std::cout << "Bending Friction Thres Theta Size: "
                  << diff_theta_t_parm_indices.size() << std::endl;

        // Yield Ori
        {
            list<IndexT> diff_yield_ori_parm_indices_buffer;
            list<IndexT> diff_yield_ori_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_yield_theta = sc.edges().find<IndexT>("diff/yield_theta_ori");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_yield_theta)
                        return;

                    for(auto&& [diff_yield_theta, E] : zip(diff_yield_theta->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_yield_theta >= 0 && init_info.valid_bending())
                        {
                            diff_yield_ori_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                            diff_yield_ori_parm_indices_buffer.push_back(diff_yield_theta);
                        }
                    }
                });

            h_diff_yield_ori_parm_indices.resize(
                diff_yield_ori_parm_indices_buffer.size());
            h_diff_yield_ori_stencil_indices.resize(
                diff_yield_ori_stencil_indices_buffer.size());

            std::ranges::move(diff_yield_ori_parm_indices_buffer,
                              h_diff_yield_ori_parm_indices.begin());
            std::ranges::move(diff_yield_ori_stencil_indices_buffer,
                              h_diff_yield_ori_stencil_indices.begin());

            diff_yield_ori_parm_indices.copy_from(h_diff_yield_ori_parm_indices);
            diff_yield_ori_stencil_indices.copy_from(h_diff_yield_ori_stencil_indices);
        }

        std::cout << "Bending Yield Theta (Ori) Size: "
                  << diff_yield_ori_parm_indices.size() << std::endl;

        // Hardening k
        {
            list<IndexT> diff_hardening_k_parm_indices_buffer;
            list<IndexT> diff_hardening_k_stencil_indices_buffer;

            info.for_each(
                geo_slots,
                [&](const ForEachInfo& info, geometry::SimplicialComplex& sc) -> auto
                {
                    auto diff_hardening_k = sc.edges().find<IndexT>("diff/hardening_k");
                    auto topo_view = sc.edges().topo().view();

                    if(!diff_hardening_k)
                        return;

                    for(auto&& [diff_hardening_k, E] : zip(diff_hardening_k->view(), topo_view))
                    {
                        auto& init_info = constitution().stencil_map[E];
                        if(diff_hardening_k >= 0 && init_info.valid_bending())
                        {
                            diff_hardening_k_stencil_indices_buffer.push_back(
                                init_info.stencil_index);
                            diff_hardening_k_parm_indices_buffer.push_back(diff_hardening_k);
                        }
                    }
                });

            h_diff_hardening_k_parm_indices.resize(
                diff_hardening_k_parm_indices_buffer.size());
            h_diff_hardening_k_stencil_indices.resize(
                diff_hardening_k_stencil_indices_buffer.size());

            std::ranges::move(diff_hardening_k_parm_indices_buffer,
                              h_diff_hardening_k_parm_indices.begin());
            std::ranges::move(diff_hardening_k_stencil_indices_buffer,
                              h_diff_hardening_k_stencil_indices.begin());

            diff_hardening_k_parm_indices.copy_from(h_diff_hardening_k_parm_indices);
            diff_hardening_k_stencil_indices.copy_from(h_diff_hardening_k_stencil_indices);
        }

        std::cout << "Bending Hardening k Size: "
                  << diff_hardening_k_parm_indices.size() << std::endl;
    }

    virtual void do_update(DiffParmUpdateInfo& info) override
    {

        //std::cout << "In Do Update" << std::endl;

        using namespace muda;

        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_kappa_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        kappa_parm_indices = diff_kappa_parm_indices.viewer().name("kappa_parm_indices"),
                        kappa_stencil_indices =
                            diff_kappa_stencil_indices.viewer().name("kappa_stencil_indices"),
                        kappas = constitution().bending_stiffnesses.viewer().name(
                            "bending_stiffness")] __device__(int I)
                       {
                           auto& kappa = kappas(kappa_stencil_indices(I));
                           kappa       = parameters(kappa_parm_indices(I));
                       });
        }

        // Threshold theta
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_theta_t_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        theta_t_parm_indices =
                            diff_theta_t_parm_indices.viewer().name("theta_t_parm_indices"),
                        theta_t_stencil_indices =
                            diff_theta_t_stencil_indices.viewer().name("theta_t_stencil_indices"),
                        thres_thetas = constitution().thres_thetas.viewer().name(
                            "thres_thetas")] __device__(int I)
                       {
                           auto& thres_theta = thres_thetas(theta_t_stencil_indices(I));
                           thres_theta = parameters(theta_t_parm_indices(I));
                       });
        }

        // Friction Stiffness
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_fs_stencil_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        fs_parm_indices = diff_fs_parm_indices.viewer().name("fs_parm_indices"),
                        fs_stencil_indices = diff_fs_stencil_indices.viewer().name("fs_stencil_indices"),
                        friciton_stiffnesses =
                            constitution().friction_stiffnesses.viewer().name(
                                "friction_stiffness")] __device__(int I)
                       {
                           auto& friction_stiffness =
                               friciton_stiffnesses(fs_stencil_indices(I));
                           friction_stiffness = parameters(fs_parm_indices(I));
                       });
        }

        // Yield Ori
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_yield_ori_parm_indices.size(),
                    [parameters = info.parameters().viewer().name("parameters"),
                     y0_parm_indices = diff_yield_ori_parm_indices.viewer().name("y0_parm_indices"),
                     y0_stencil_indices =
                         diff_yield_ori_stencil_indices.viewer().name("y0_stencil_indices"),
                     yield_ori_thetas = constitution().yield_ori_thetas.viewer().name(
                         "yield_ori")] __device__(int I)
                    {
                        auto& yield_ori_theta = yield_ori_thetas(y0_stencil_indices(I));
                        yield_ori_theta = parameters(y0_parm_indices(I));
                    });
        }

        // Hardening k
        {
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_hardening_k_parm_indices.size(),
                       [parameters = info.parameters().viewer().name("parameters"),
                        kh_parm_indices = diff_hardening_k_parm_indices.viewer().name("kh_parm_indices"),
                        kh_stencil_indices =
                            diff_hardening_k_stencil_indices.viewer().name("kh_stencil_indices"),
                        hardening_ks = constitution().hardening_ks.viewer().name(
                            "hardening_k")] __device__(int I)
                       {
                           auto& hardening_k = hardening_ks(kh_stencil_indices(I));
                           hardening_k = parameters(kh_parm_indices(I));
                       });
        }
    }

    virtual void do_report_extent(DiffParmExtentInfo& info) override
    {
        //std::cout << "In Do Report extent" << std::endl;

        info.triplet_count(4 * diff_kappa_stencil_indices.size()
                           + 4 * diff_fs_stencil_indices.size()
                           + 4 * diff_theta_t_stencil_indices.size()
                           + 4 * diff_yield_ori_stencil_indices.size()
                           + 4 * diff_hardening_k_stencil_indices.size());
    }

    virtual void do_assemble(DiffParmInfo& info) override
    {
        using namespace muda;
        namespace DSGFPB = sym::discrete_shell_friction_hardening_plastic_bending;

        auto pGpPs = info.pGpP();

        // assamble kappa
        {
            SizeT bending_k_offset = 0;

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_kappa_stencil_indices.size(),
                    [diff_kappa_stencil = diff_kappa_stencil_indices.viewer().name("diff_kappa_stencil"),
                     diff_kappa_parm_indices = diff_kappa_parm_indices.viewer().name("kappa_parm_indices"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     theta_bars = constitution().theta_bars.viewer().name("theta_bar"),
                     bending_stiffnesses =
                         constitution().bending_stiffnesses.viewer().name("bending_stiffnesses"),
                     plastic_thetas = info.plastic_thetas().viewer().name("plastic_theta"),
                     d_plastic_theta_d_ks = info.d_plastic_theta_d_k().viewer().name("d_plastic_theta_d_k"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_k_offset] __device__(int I) mutable
                    {
                        IndexT stencils_index = diff_kappa_stencil(I);
                        IndexT parm_index     = diff_kappa_parm_indices(I);

                        Vector4i idx = stencils(stencils_index);

                        Float L0            = L0s(stencils_index);
                        Float h_bar         = h_bars(stencils_index);
                        Float plastic_theta = plastic_thetas(stencils_index);
                        Float theta_bar     = theta_bars(stencils_index);
                        Float V_bar         = V_bars(stencils_index);

                        Float bending_stiffness = bending_stiffnesses(stencils_index);
                        Float d_plastic_theta_d_k =
                            d_plastic_theta_d_ks(stencils_index);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt;

                        Vector12 pGpKappa;
                        DSGFPB::pGpKappa(
                            pGpKappa, x0, x1, x2, x3, L0, h_bar, bending_stiffness, theta_bar, plastic_theta, d_plastic_theta_d_k);

                        pGpKappa *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpKappa, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_index;

                            pGpPs(bending_k_offset + I * 4 + k)
                                .write(vI, J, pGpKappa.segment<3>(k * 3));
                        }
                    });
        }


        // Friction Stiffness
        {
            SizeT bending_friction_k_offset = 4 * diff_kappa_stencil_indices.size();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_fs_stencil_indices.size(),
                       [diff_fs_stencil = diff_fs_stencil_indices.viewer().name("diff_fs_stencil"),
                        diff_fs_parm_indices = diff_fs_parm_indices.viewer().name("fsparm_indices"),
                        stencils = constitution().stencils.viewer().name("stencils"),
                        theta_bars = constitution().theta_bars.viewer().name("theta_bar"),
                        anchor_thetas = info.anchor_thetas().viewer().name("anchor_theta"),
                        thicknesses = info.thicknesses().viewer().name("thicknesses"),
                        h_bars = constitution().h_bars.viewer().name("h_bar"),
                        V_bars = constitution().V_bars.viewer().name("V_bar"),
                        L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                        xs       = info.xs().viewer().name("xs"),
                        is_fixed = info.is_fixed().viewer().name("is_fixed"),
                        pGpPs    = pGpPs.viewer().name("pGpPs"),
                        dt       = info.dt(),
                        bending_friction_k_offset] __device__(int I) mutable
                       {
                           IndexT stencils_index = diff_fs_stencil(I);
                           IndexT parm_index     = diff_fs_parm_indices(I);

                           Vector4i idx          = stencils(stencils_index);
                           Float    L0           = L0s(stencils_index);
                           Float    h_bar        = h_bars(stencils_index);
                           Float    theta_bar    = theta_bars(stencils_index);
                           Float    anchor_theta = anchor_thetas(stencils_index);
                           Float    V_bar        = V_bars(stencils_index);

                           Vector3 x0 = xs(idx[0]);
                           Vector3 x1 = xs(idx[1]);
                           Vector3 x2 = xs(idx[2]);
                           Vector3 x3 = xs(idx[3]);

                           Float Vdt2 = V_bar * dt * dt;

                           Vector12 pGpKappa;
                           DSGFPB::pGpFrictionStiffness(
                               pGpKappa, x0, x1, x2, x3, L0, h_bar, theta_bar, anchor_theta);

                           pGpKappa *= Vdt2;

                           Vector4i ignore = {is_fixed(idx[0]),
                                              is_fixed(idx[1]),
                                              is_fixed(idx[2]),
                                              is_fixed(idx[3])};

                           zero_out(pGpKappa, ignore);

                           for(int k = 0; k < 4; ++k)  // for each vertex
                           {
                               auto vI = idx(k);
                               auto J  = parm_index;

                               pGpPs(bending_friction_k_offset + I * 4 + k)
                                   .write(vI, J, pGpKappa.segment<3>(k * 3));
                           }
                       });
        }

        // assumble theta_thres
        {
            SizeT bending_thres_offset = 4 * diff_kappa_stencil_indices.size()
                                         + 4 * diff_fs_stencil_indices.size();
            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(diff_theta_t_stencil_indices.size(),
                       [diff_theta_t_stencil =
                            diff_theta_t_stencil_indices.viewer().name("diff_theta_t_stencil"),
                        diff_theta_t_parm_indices =
                            diff_theta_t_parm_indices.viewer().name("theta_t_parm_indices"),
                        stencils = constitution().stencils.viewer().name("stencils"),
                        friction_stiffnesses =
                            constitution().friction_stiffnesses.viewer().name("friction_stiffness"),
                        theta_bars = constitution().theta_bars.viewer().name("theta_bar"),
                        d_anchor_thetas_d_thres_thetas =
                            info.d_anchor_thetas_d_thres_theta().viewer().name("d_anchor_thetas_d_thres_theta"),
                        thicknesses = info.thicknesses().viewer().name("thicknesses"),
                        h_bars = constitution().h_bars.viewer().name("h_bar"),
                        V_bars = constitution().V_bars.viewer().name("V_bar"),
                        L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                        xs       = info.xs().viewer().name("xs"),
                        is_fixed = info.is_fixed().viewer().name("is_fixed"),
                        pGpPs    = pGpPs.viewer().name("pGpPs"),
                        dt       = info.dt(),
                        bending_thres_offset] __device__(int I) mutable
                       {
                           IndexT stencils_index = diff_theta_t_stencil(I);
                           IndexT parm_index     = diff_theta_t_parm_indices(I);

                           Vector4i idx = stencils(stencils_index);

                           Float friction_stiffness =
                               friction_stiffnesses(stencils_index);
                           Float d_anchor_theta_d_thres_theta =
                               d_anchor_thetas_d_thres_thetas(stencils_index);
                           Float L0        = L0s(stencils_index);
                           Float h_bar     = h_bars(stencils_index);
                           Float theta_bar = theta_bars(stencils_index);
                           Float V_bar     = V_bars(stencils_index);

                           Vector3 x0 = xs(idx[0]);
                           Vector3 x1 = xs(idx[1]);
                           Vector3 x2 = xs(idx[2]);
                           Vector3 x3 = xs(idx[3]);

                           Float Vdt2 = V_bar * dt * dt;

                           Vector12 pGpThetaY;
                           DSGFPB::pGpThetaThres(
                               pGpThetaY, x0, x1, x2, x3, L0, h_bar, d_anchor_theta_d_thres_theta, friction_stiffness);

                           pGpThetaY *= Vdt2;

                           Vector4i ignore = {is_fixed(idx[0]),
                                              is_fixed(idx[1]),
                                              is_fixed(idx[2]),
                                              is_fixed(idx[3])};

                           zero_out(pGpThetaY, ignore);

                           for(int k = 0; k < 4; ++k)  // for each vertex
                           {
                               auto vI = idx(k);
                               auto J  = parm_index;

                               pGpPs(bending_thres_offset + I * 4 + k)
                                   .write(vI, J, pGpThetaY.segment<3>(k * 3));
                           }
                       });
        }


        // assamble yield_ori
        {
            SizeT bending_yield_ori_offset = 4 * diff_kappa_stencil_indices.size()
                                             + 4 * diff_fs_stencil_indices.size()
                                             + 4 * diff_theta_t_stencil_indices.size();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_yield_ori_stencil_indices.size(),
                    [diff_yield_ori_stencil =
                         diff_yield_ori_stencil_indices.viewer().name("diff_yield_ori_stencil"),
                     diff_yield_ori_parm_indices =
                         diff_yield_ori_parm_indices.viewer().name("yield_ori_parm_indices"),
                     bending_stiffnesses =
                         constitution().bending_stiffnesses.viewer().name("bending_stiffness"),
                     d_plastic_theta_d_yield_oris =
                         info.d_plastic_theta_d_yield_ori().viewer().name("d_plastic_theta_d_yield_ori"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     theta_bars = constitution().theta_bars.viewer().name("theta_bar"),
                     plastic_thetas = info.plastic_thetas().viewer().name("plastic_theta"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_yield_ori_offset] __device__(int I) mutable
                    {
                        IndexT stencils_index = diff_yield_ori_stencil(I);
                        IndexT parm_index     = diff_yield_ori_parm_indices(I);

                        Vector4i idx = stencils(stencils_index);

                        Float L0            = L0s(stencils_index);
                        Float h_bar         = h_bars(stencils_index);
                        Float plastic_theta = plastic_thetas(stencils_index);
                        Float theta_bar     = theta_bars(stencils_index);
                        Float V_bar         = V_bars(stencils_index);

                        Float bending_stiffness =
                            bending_stiffnesses(stencils_index);
                        Float d_plastic_theta_d_yield_ori =
                            d_plastic_theta_d_yield_oris(stencils_index);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt;

                        Vector12 pGpYieldOri;
                        DSGFPB::pGpYield(pGpYieldOri, x0, x1, x2, x3, L0, h_bar, bending_stiffness, d_plastic_theta_d_yield_ori);

                        pGpYieldOri *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpYieldOri, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_index;

                            pGpPs(bending_yield_ori_offset + I * 4 + k)
                                .write(vI, J, pGpYieldOri.segment<3>(k * 3));
                        }
                    });
        }

        // assamble hardening_k
        {
            SizeT bending_harndeing_k_offset =
                4 * diff_kappa_stencil_indices.size()
                + 4 * diff_fs_stencil_indices.size()
                + 4 * diff_theta_t_stencil_indices.size()
                + 4 * diff_yield_ori_stencil_indices.size();

            ParallelFor()
                .file_line(__FILE__, __LINE__)
                .apply(
                    diff_hardening_k_stencil_indices.size(),
                    [diff_hardening_k_stencil =
                         diff_hardening_k_stencil_indices.viewer().name("diff_hardening_k_stencil"),
                     diff_hardening_k_parm_indices =
                         diff_hardening_k_parm_indices.viewer().name("hardening_k_parm_indices"),
                     bending_stiffnesses =
                         constitution().bending_stiffnesses.viewer().name("bending_stiffness"),
                     d_plastic_theta_d_hardening_ks =
                         info.d_plastic_theta_d_hardening_k().viewer().name("d_plastic_theta_d_hardening_k"),
                     stencils = constitution().stencils.viewer().name("stencils"),
                     theta_bars = constitution().theta_bars.viewer().name("theta_bar"),
                     plastic_thetas = info.plastic_thetas().viewer().name("plastic_theta"),
                     thicknesses = info.thicknesses().viewer().name("thicknesses"),
                     h_bars = constitution().h_bars.viewer().name("h_bar"),
                     V_bars = constitution().V_bars.viewer().name("V_bar"),
                     L0s = constitution().rest_lengths.viewer().name("rest_lengths"),
                     xs       = info.xs().viewer().name("xs"),
                     is_fixed = info.is_fixed().viewer().name("is_fixed"),
                     pGpPs    = pGpPs.viewer().name("pGpPs"),
                     dt       = info.dt(),
                     bending_harndeing_k_offset] __device__(int I) mutable
                    {
                        IndexT stencils_index = diff_hardening_k_stencil(I);
                        IndexT parm_index = diff_hardening_k_parm_indices(I);

                        Vector4i idx = stencils(stencils_index);

                        Float L0            = L0s(stencils_index);
                        Float h_bar         = h_bars(stencils_index);
                        Float plastic_theta = plastic_thetas(stencils_index);
                        Float theta_bar     = theta_bars(stencils_index);
                        Float V_bar         = V_bars(stencils_index);

                        Float bending_stiffness =
                            bending_stiffnesses(stencils_index);
                        Float d_plastic_theta_d_hardening_k =
                            d_plastic_theta_d_hardening_ks(stencils_index);

                        Vector3 x0 = xs(idx[0]);
                        Vector3 x1 = xs(idx[1]);
                        Vector3 x2 = xs(idx[2]);
                        Vector3 x3 = xs(idx[3]);

                        Float Vdt2 = V_bar * dt * dt;

                        Vector12 pGpYieldOri;
                        DSGFPB::pGpYield(pGpYieldOri, x0, x1, x2, x3, L0, h_bar, bending_stiffness, d_plastic_theta_d_hardening_k);

                        pGpYieldOri *= Vdt2;

                        Vector4i ignore = {is_fixed(idx[0]),
                                           is_fixed(idx[1]),
                                           is_fixed(idx[2]),
                                           is_fixed(idx[3])};

                        zero_out(pGpYieldOri, ignore);

                        for(int k = 0; k < 4; ++k)  // for each vertex
                        {
                            auto vI = idx(k);
                            auto J  = parm_index;

                            pGpPs(bending_harndeing_k_offset + I * 4 + k)
                                .write(vI, J, pGpYieldOri.segment<3>(k * 3));
                        }
                    });
        }
    }
};

REGISTER_SIM_SYSTEM(DiffDiscreteShellFrictionHardeningPlasticBending);
}  // namespace uipc::backend::cuda
