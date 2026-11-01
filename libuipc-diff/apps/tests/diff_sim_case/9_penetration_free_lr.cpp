#include <catch.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <uipc/diff_sim/learning_rate_feature.h>
#include "diff_sim_solver/diff_sim_solver.h"

TEST_CASE("9_penetration_free_lr", "[abd]")
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
    auto learning_rate_feature = world.features().find<diff_sim::LearningRateFeature>();
    REQUIRE(learning_rate_feature != nullptr);
    auto update_parameter_feature =
        world.features().find<diff_sim::UpdateParameterFeature>();

    world.dump();

    constexpr IndexT EndFrame = 1;

    world.recover(0);
    VectorX U;
    U.resize(1);
    U(0)        = 0.2;
    Float alpha = learning_rate_feature->learning_rate(from_eigen(U));
    REQUIRE(alpha < 1);
    U(0) = -0.9;
    update_parameter_feature->update(from_eigen(U));
    world.retrieve();

    sio.write_surface(
        fmt::format("{}/scene_surface{}.obj", this_output_path, world.frame()));

    while(world.frame() < EndFrame)
    {
        world.advance();
        world.retrieve();
        sio.write_surface(
            fmt::format("{}/scene_surface{}.obj", this_output_path, world.frame()));
    }
}