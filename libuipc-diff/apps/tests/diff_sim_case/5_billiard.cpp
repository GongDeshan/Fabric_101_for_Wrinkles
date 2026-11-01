#include <catch.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/particle.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include "diff_sim_solver/diff_sim_solver.h"

TEST_CASE("5_billiard", "[fem]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    namespace fs = std::filesystem;

    std::string tetmesh_dir{AssetDir::tetmesh_path()};
    auto        this_output_path = AssetDir::output_path(__FILE__);

    Engine engine{"cuda", this_output_path};
    World  world{engine};

    auto config = Scene::default_config();

    config["gravity"]                       = Vector3{0, 0, 0};
    config["contact"]["enable"]             = true;
    config["contact"]["d_hat"]              = 0.005;
    config["contact"]["friction"]["enable"] = true;
    config["line_search"]["max_iter"]       = 8;
    config["linear_system"]["tol_rate"]     = 1e-3;

    {  // dump config
        std::ofstream ofs(fmt::format("{}config.json", this_output_path));
        ofs << config.dump(4);
    }

    SimplicialComplexIO io;

    Scene scene{config};
    scene.contact_tabular().default_model(0.0, 0.1_GPa);
    auto ball_ce = scene.contact_tabular().create("balls");
    scene.contact_tabular().insert(ball_ce, ball_ce, 0.5, 0.1_GPa);
    // create constitution and contact model
    Particle pt;

    // create object
    Float weight   = 0.170;  // 170g
    Float diameter = 0.055;  // 55mm
    Float density = weight / (4.0 / 3.0 * std::numbers::pi * std::pow(diameter / 2, 3));

    auto object = scene.objects().create("balls");

    vector<Vector3> Vs = {
        Vector3{0, 0, 0},
        Vector3{0.4, 0, 0},
    };

    auto mesh = pointcloud(Vs);

    label_surface(mesh);

    pt.apply_to(mesh, density, diameter / 2);

    auto vel = mesh.vertices().create<Vector3>(builtin::velocity, Vector3::Zero());
    auto diff_vel = mesh.vertices().create<Vector3i>("diff/velocity", -Vector3i::Ones());
    {
        auto diff_vel_view = view(*diff_vel);
        diff_vel_view[0]   = Vector3i{0, -1, 1};  // x, y, z of the first vertex
    }
    ball_ce.apply_to(mesh);
    auto [geo_slot, rest_geo_slot] = object->geometries().create(mesh);

    auto ground_obj = scene.objects().create("ground");
    auto g          = ground();
    // ground_obj->geometries().create(g);


    auto& diff_sim = scene.diff_sim();
    auto& parms    = diff_sim.parameters();
    parms.resize(2);  // 2 parameters for init velocity (x,z) of particle

    world.init(scene);


    SceneIO sio{scene};
    sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, 0));

    IndexT  EndFrame = 20;
    Vector2 aim_pos{2, 0};  // x z
    Float   K = 50;

    using test_case::DiffSimSolver;

    DiffSimSolver solver{world, this_output_path};
    solver.total_frame_count(EndFrame);
    solver.save_surf(true);

    solver.U().resize(2);
    solver.U().segment<2>(0) = Vector2{0, 0};

    spdlog::set_level(spdlog::level::warn);

    solver.SelectDofs(
        [&](DiffSimSolver::SelectDofInfo& info)
        {
            if(info.frame() < EndFrame)
                return;

            auto& geo = geo_slot->geometry();

            auto offset   = geo.meta().find<IndexT>(builtin::dof_offset);
            auto offset_v = offset->view()[0];

            auto& sdi = info.sdi();

            sdi.resize(2);
            sdi(0) = offset_v + 3;
            sdi(1) = offset_v + 5;
        });

    solver.L(
        [&](DiffSimSolver::ComputeLossInfo& info) -> Float
        {
            auto& X = info.X();
            return 0.5 * K * (X - aim_pos).squaredNorm();
        });

    solver.dLdX(
        [&](DiffSimSolver::ComputedLdXInfo& info)
        {
            auto& X    = info.X();
            auto& dLdX = info.dLdX();
            dLdX       = K * (X - aim_pos);
        });

    solver.ExitCondition(
        [&](DiffSimSolver::ExitConditionInfo& info) -> bool
        {
            auto& dLdU = info.dLdU();

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