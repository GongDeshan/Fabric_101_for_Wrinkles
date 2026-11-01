#include <catch.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/particle.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include "diff_sim_solver/diff_sim_solver.h"

TEST_CASE("0_particle_diff_init_velocity", "[fem]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    namespace fs = std::filesystem;

    spdlog::set_level(spdlog::level::warn);

    std::string tetmesh_dir{AssetDir::tetmesh_path()};
    auto        this_output_path = AssetDir::output_path(__FILE__);


    Engine engine{"cuda", this_output_path};
    World  world{engine};

    auto config = Scene::default_config();

    config["gravity"]                   = Vector3{0, -9.8, 0};
    config["contact"]["enable"]         = false;  // disable contact
    config["line_search"]["max_iter"]   = 8;
    config["linear_system"]["tol_rate"] = 1e-3;

    {  // dump config
        std::ofstream ofs(fmt::format("{}config.json", this_output_path));
        ofs << config.dump(4);
    }

    SimplicialComplexIO io;

    Scene scene{config};
    auto& diff_sim = scene.diff_sim();
    auto& parms    = diff_sim.parameters();
    parms.resize(2);  // 2 parameters for init velocity of particle

    // create constitution and contact model
    Particle pt;

    // create object
    auto object = scene.objects().create("point");

    vector<Vector3> Vs = {Vector3{0, 0, 0}};

    auto mesh = pointcloud(Vs);

    label_surface(mesh);
    pt.apply_to(mesh, 1, 0.0);

    auto vel = mesh.vertices().create<Vector3>(builtin::velocity, Vector3::Zero());
    auto diff_vel = mesh.vertices().create<Vector3i>("diff/velocity", {0, 1, -1});  // mask-off the z component
    auto [geo_slot, rest_geo_slot] = object->geometries().create(mesh);
    world.init(scene);
    world.dump();  // dump, later we can recover from this dump

    SceneIO sio{scene};
    sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, 0));

    Float k = 10;

    auto dLossdX = [&](const Vector3& x) -> Vector3 {
        return -k * Vector3{0, 1, 0};
    };

    IndexT EndFrame = 10;

    using test_case::DiffSimSolver;

    DiffSimSolver solver{world, this_output_path};
    solver.total_frame_count(EndFrame);
    solver.save_surf(true);


    auto& U = solver.U();
    U.resize(1);  // 1 parameter for theta
    Float& theta = U[0];
    theta        = 0.1 * std::numbers::pi;

    solver.SelectDofs(
        [&](DiffSimSolver::SelectDofInfo& info)
        {
            if(info.frame() < EndFrame)
                return;

            auto& geo          = geo_slot->geometry();
            auto  dof_offset   = geo.meta().find<IndexT>(builtin::dof_offset);
            auto  dof_offset_v = dof_offset->view()[0];

            auto& sdi = info.sdi();

            sdi.resize(1);
            sdi(0) = dof_offset_v + 1;  // y component
        });

    solver.L(
        [&](DiffSimSolver::ComputeLossInfo& info) -> Float
        {
            auto& X = info.X();
            return -k * X(0);
        });

    solver.dLdX(
        [&](DiffSimSolver::ComputedLdXInfo& info)
        {
            auto& dLdX = info.dLdX();

            dLdX.resize(1);
            dLdX(0) = -k;
        });

    solver.PofU(
        [](DiffSimSolver::ComputePofUInfo& info)
        {
            auto& P = info.P();
            auto& U = info.U();

            P = 5 * Vector2{std::cos(U[0]), std::sin(U[0])};
        });

    solver.dPdU(
        [](DiffSimSolver::ComputedPdUInfo& info)
        {
            auto& dPdU = info.dPdU();
            auto& U    = info.U();

            dPdU.reserve(2);
            dPdU.insert(0, 0) = -5 * std::sin(U[0]);
            dPdU.insert(1, 0) = 5 * std::cos(U[0]);
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


    std::cout << "Final theta: " << theta << "\n";
}