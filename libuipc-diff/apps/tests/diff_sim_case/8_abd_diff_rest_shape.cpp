#include <catch.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include "diff_sim_solver/diff_sim_solver.h"

TEST_CASE("8_abd_diff_rest_shape", "[abd]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    namespace fs = std::filesystem;

    std::string trimesh_dir{AssetDir::trimesh_path()};
    std::string tetmesh_dir{AssetDir::tetmesh_path()};
    auto        this_output_path = AssetDir::output_path(__FILE__);


    Engine engine{"cuda", this_output_path};
    World  world{engine};

    auto config = Scene::default_config();

    config["gravity"]                       = Vector3{0, -9.8, 0};
    config["dt"]                            = 1;
    config["contact"]["enable"]             = true;
    config["contact"]["d_hat"]              = 0.1;
    config["contact"]["friction"]["enable"] = false;
    config["line_search"]["max_iter"]       = 8;
    config["linear_system"]["tol_rate"]     = 1e-3;
    config["newton"]["velocity_tol"]        = 0.05;

    {  // dump config
        std::ofstream ofs(fmt::format("{}config.json", this_output_path));
        ofs << config.dump(4);
    }

    Transform           pre_transform = Transform::Identity();
    SimplicialComplexIO io{pre_transform};

    Scene scene{config};
    auto& diff_sim = scene.diff_sim();
    auto& parms    = diff_sim.parameters();
    scene.contact_tabular().default_model(0.0, 1e6);

    // create constitution and contact model
    AffineBodyConstitution abd;

    // create object
    auto object = scene.objects().create("cube");

    vector<Vector3> Vs = {Vector3{0, -1, 0},
                          Vector3{0, 0, 1},
                          Vector3{std::sqrt(3) / 2, 0, -0.5},
                          Vector3{-std::sqrt(3) / 2, 0, -0.5}};

    vector<Vector4i> Ts = {Vector4i{0, 1, 2, 3}};

    auto tet = tetmesh(Vs, Ts);
    label_surface(tet);
    label_triangle_orient(tet);
    auto mesh0 = extract_surface(tet);
    auto mesh  = flip_inward_triangles(mesh0);

    abd.apply_to(mesh, 100.0_MPa);
    auto diff_pos = mesh.vertices().create<Vector3i>("diff/position", -Vector3i::Ones());
    auto diff_pos_view = view(*diff_pos);
    auto pos_view      = view(mesh.positions());

    vector<IndexT> vs = {0};
    parms.resize(vs.size());  // take y as the parameter
    auto parm_view = as_eigen(parms);
    for(auto&& [i, vI] : enumerate(vs))
    {
        diff_pos_view[vI] = Vector3i{-1, (IndexT)i, -1};
        parm_view[i]      = pos_view[vI].y();
    }

    auto [geo_slot, rest_geo_slot] = object->geometries().create(mesh, mesh);

    auto ground_obj = scene.objects().create("ground");
    auto g          = ground(-1.1);
    ground_obj->geometries().create(g);

    SceneIO sio{scene};
    world.init(scene);
    world.dump();

    constexpr IndexT EndFrame = 1;

    sio.write_surface(
        fmt::format("{}/scene_surface{}.obj", this_output_path, world.frame()));
    while(world.frame() < EndFrame)
    {
        world.advance();
        world.retrieve();
        sio.write_surface(
            fmt::format("{}/scene_surface{}.obj", this_output_path, world.frame()));
    }

    world.recover(0);

    spdlog::set_level(spdlog::level::warn);


    Float           aim_height = -0.5;
    constexpr Float K          = 1;

    using test_case::DiffSimSolver;

    DiffSimSolver solver{world, this_output_path};
    solver.save_surf(true);
    solver.total_frame_count(EndFrame);
    solver.max_line_search(8);

    auto& U = solver.U();
    U.resize(parms.size());
    U = parm_view;

    solver.SelectDofs(
        [&](DiffSimSolver::SelectDofInfo& info)
        {
            if(info.frame() < EndFrame)
                return;

            auto& geo = geo_slot->geometry();

            auto offset   = geo.meta().find<IndexT>(builtin::dof_offset);
            auto offset_v = offset->view()[0];

            auto& sdi = info.sdi();

            sdi.resize(1);
            sdi(0) = offset_v + 1;  // y
        });


    solver.L(
        [&](DiffSimSolver::ComputeLossInfo& info) -> Float
        {
            auto& X = info.X();
            Float L = 0.5 * K * std::pow(X[0] - aim_height, 2);
            return L;
        });

    solver.dLdX(
        [&](DiffSimSolver::ComputedLdXInfo& info)
        {
            auto& X    = info.X();
            auto& dLdX = info.dLdX();
            dLdX[0]    = K * (X[0] - aim_height);
        });


    solver.ExitCondition(
        [&](DiffSimSolver::ExitConditionInfo& info) -> bool
        {
            auto& geo = geo_slot->geometry();
            auto  P   = geo.positions().view();
            auto  v   = geo.transforms().view()[0];
            std::cout << "Pos:" << P[0] << std::endl;

            auto& dLdU = info.dLdU();
            std::cout << "dLdU:" << dLdU.transpose() << std::endl;

            auto norm = dLdU.lpNorm<Eigen::Infinity>();
            std::cout << "dLdU Norm: " << norm << std::endl;
            auto rel_loss = info.rel_loss();
            std::cout << "rel loss: " << rel_loss << std::endl;
            std::cout << "X:" << info.X().transpose() << std::endl;
            return norm < 1e-4 || rel_loss < 1e-4;
        });

    // should converge in 15 iterations
    REQUIRE(solver.gradient_descent(15));
}