#include <catch.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/particle.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include "diff_sim_solver/diff_sim_solver.h"

TEST_CASE("7_diff_friction_rate", "[fem]")
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

    config["gravity"]                       = Vector3{0, -10, 0};
    config["contact"]["enable"]             = true;
    config["contact"]["d_hat"]              = 0.1;
    config["contact"]["friction"]["enable"] = true;
    config["line_search"]["max_iter"]       = 8;
    config["linear_system"]["tol_rate"]     = 1e-3;

    {  // dump config
        std::ofstream ofs(fmt::format("{}config.json", this_output_path));
        ofs << config.dump(4);
    }

    Scene scene{config};

    auto& diff_sim = scene.diff_sim();
    auto& parms    = diff_sim.parameters();
    parms.resize(1);  // one parameter for friction_rate

    auto& ct             = scene.contact_tabular();
    auto  friction_rates = ct.contact_models().find<Float>("friction_rate");
    auto  diff_friction_rate =
        ct.contact_models().create<IndexT>("diff/friction_rate", -1);  // -1 means ignore
    view(*diff_friction_rate)[0] = 0;  // the friction_rate mapping to the 0th parameter.

    // create constitution and contact model
    Particle pt;

    // create object
    auto object = scene.objects().create("point");

    vector<Vector3> Vs = {Vector3{0, 0.1, 0}};

    auto mesh = pointcloud(Vs);

    label_surface(mesh);
    pt.apply_to(mesh, 10.0 /*KG*/, 0.0);
    auto vel = mesh.vertices().create<Vector3>(builtin::velocity, Vector3::Zero());
    view(*vel)[0]                  = Vector3{6.0, 0, 0};
    auto [geo_slot, rest_geo_slot] = object->geometries().create(mesh);

    auto ground_obj = scene.objects().create("ground");
    auto g          = ground();
    ground_obj->geometries().create(g);

    world.init(scene);

    auto  parm_vector        = as_eigen(parms);
    Float init_friction_rate = 0.0;
    parm_vector[0]           = init_friction_rate;

    // broadcast the parameter
    parms.broadcast();

    SceneIO sio{scene};
    sio.write_surface(fmt::format("{}scene_surface{}.obj", this_output_path, 0));

    IndexT          EndFrame = 20;
    Vector3         aim_pos{1, 0};
    constexpr Float K = 50.0;

    using test_case::DiffSimSolver;

    DiffSimSolver solver{world, this_output_path};
    solver.total_frame_count(EndFrame);
    solver.save_surf(true);

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
            sdi(0) = offset_v + 0;
            sdi(1) = offset_v + 2;
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