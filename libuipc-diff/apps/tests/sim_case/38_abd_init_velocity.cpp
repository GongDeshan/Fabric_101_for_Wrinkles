#include <catch.hpp>
#include <app/asset_dir.h>
#include <uipc/uipc.h>
#include <uipc/constitution/affine_body_constitution.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>

TEST_CASE("38_abd_init_velocity", "[fem]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::geometry;
    using namespace uipc::constitution;

    namespace fs = std::filesystem;

    std::string trimesh_dir{AssetDir::trimesh_path()};
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

    Transform pre_transform = Transform::Identity();
    pre_transform.scale(0.3);
    SimplicialComplexIO io{pre_transform};

    Scene scene{config};
    auto& diff_sim = scene.diff_sim();
    auto& parms    = diff_sim.parameters();
    parms.resize(3);  // 3 parameters for init velocity of particle

    // create constitution and contact model
    AffineBodyConstitution abd;

    // create object
    auto object = scene.objects().create("cube");
    {
        vector<Vector3> Vs = {Vector3{0, 0, 0}};

        auto mesh = io.read(fmt::format("{}cube.obj", trimesh_dir));

        label_surface(mesh);
        abd.apply_to(mesh, 100.0_MPa);
        auto      vel = mesh.instances().create<Matrix4x4>(builtin::velocity);
        Transform t   = Transform::Identity();
        t.translate(Vector3{2, 2, 0});
        view(*vel)[0] = t.matrix();

        object->geometries().create(mesh);
    }

    auto ref_object = scene.objects().create("ref_cube");
    {
        vector<Vector3> Vs = {Vector3{0, 0, 0}};

        auto mesh = io.read(fmt::format("{}cube.obj", trimesh_dir));

        label_surface(mesh);
        abd.apply_to(mesh, 100.0_MPa);

        Transform t = Transform::Identity();
        t.translate(Vector3{-0.5, 0, 0});
        view(mesh.transforms())[0] = t.matrix();
        ref_object->geometries().create(mesh);
    }

    SceneIO sio{scene};
    world.init(scene);

    sio.write_surface(
        fmt::format("{}scene_surface{}.obj", this_output_path, world.frame()));

    while(world.frame() < 50)
    {
        world.advance();
        world.retrieve();
        sio.write_surface(
            fmt::format("{}scene_surface{}.obj", this_output_path, world.frame()));
    }
}