"""Fold the fitted Paper mesh, launch it, and apply an independent wind load."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from uipc import Logger, builtin, view
from uipc.constitution import (
    DiscreteShellFrictionHardeningPlasticBending,
    SoftPositionConstraint,
    StVKNonlinearFrictionHardeningPlasticShell,
)
from uipc.core import Engine, Scene, World
from uipc.geometry import SimplicialComplexIO, label_surface

from load_mesh import load_mesh_uv


DT = 0.01
CREASE_FRAMES = 49
SHAPE_FRAMES = 51
THROW_FRAMES = 10
FOLD_END = CREASE_FRAMES + SHAPE_FRAMES
RELEASE = FOLD_END + THROW_FRAMES + 1

MASS_DENSITY = 992.875
THICKNESS = 0.00008

# Fixed isotropic stretching stiffness and learned shear parameters, in Pa.
Warp_Stiffness = 188180336.22351965
Weft_Stiffness = 188180336.22351965
K11 = Warp_Stiffness
K22 = Weft_Stiffness
K12 = 0.0

K33_0 = 9874486.76382516
K33_1 = 11047956.090603216
Fric_K33 = 11993285.28583167
Hardening_K33 = 104082.21194229477
Thres_uv = 0.008529096038169046
Yield_UV = 0.021834275670439288

# Best isotropic bending checkpoint from warp five-parameter training.
BENDING_STIFFNESS = 45680.40496350644
BENDING_FRICTION = 41512.9186031141
THRES_THETA = 0.18225525128754413
Hardening_K = 1003.7625913305119
YIELD_THETA = 0.29320661294010936
PLASTIC_THETA = 0.0
ANCHOR_THETA = 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--wind", type=float, nargs=3, default=(0.0, 4.0, -3.0),
                        metavar=("VX", "VY", "VZ"), help="Air velocity in world coordinates (m/s).")
    parser.add_argument("--no-wind", action="store_true", help="Disable aerodynamic forces.")
    parser.add_argument("--launch-speed", type=float, default=3.0, help="Launch speed toward -Z (m/s).")
    parser.add_argument("--launch-angle", type=float, default=12.0,
                        help="Launch elevation in degrees above the horizontal -Z direction.")
    default_assets = Path(__file__).resolve().parent / "meshes"
    parser.add_argument("--assets", type=Path, default=default_assets)
    args = parser.parse_args()
    if args.frames < 1 or not np.isfinite([*args.wind, args.launch_speed]).all() or args.launch_speed < 0:
        parser.error("Use positive frame count, finite wind and nonnegative launch speed.")

    if not -90 < args.launch_angle < 90:
        parser.error("Use a launch angle between -90 and 90 degrees.")
    angle = np.deg2rad(args.launch_angle)
    launch_velocity = args.launch_speed * np.array([0.0, np.sin(angle), -np.cos(angle)])

    out = (Path(__file__).resolve().parent.parent / "animation_save_path" /
           datetime.now().strftime("fold_throw_paper_tailwind_%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True)
    print("Mesh output:", out, flush=True)

    Logger.set_level(Logger.Warn)
    config = Scene.default_config()
    config["dt"] = DT
    config["contact"]["enable"] = True
    config["contact"]["d_hat"] = 0.001
    config["diff_sim"]["enable"] = False
    scene = Scene(config)
    io = SimplicialComplexIO()
    raw_path = args.assets / "paper_plane_raw.obj"
    mesh = io.read(str(raw_path))
    folded = io.read(str(args.assets / "paper_plane_final.obj"))
    rest = np.asarray(mesh.positions().view()).reshape(-1, 3).copy()
    target = np.asarray(folded.positions().view()).reshape(-1, 3).copy()
    faces = np.asarray(mesh.triangles().topo().view()).reshape(-1, 3)
    final_faces = np.asarray(folded.triangles().topo().view()).reshape(-1, 3)
    if rest.shape != target.shape or not np.array_equal(
        np.unique(np.sort(faces, axis=1), axis=0),
        np.unique(np.sort(final_faces, axis=1), axis=0),
    ):
        raise ValueError("Folding requires matching raw and folded vertex indices and faces.")

    label_surface(mesh)
    StVKNonlinearFrictionHardeningPlasticShell().apply_to(
        mesh, K11, K12, K22, K33_0, K33_1,
        hardening_k33=Hardening_K33, friction_k33=Fric_K33,
        thres_uv=Thres_uv, anchor_strain_uv=0.0, yield_uv=Yield_UV,
        plastic_strain_uv=0.0, mass_density=MASS_DENSITY, thickness=THICKNESS,
    )
    view(mesh.triangles().find("rest_x")).reshape(-1, 6)[:] = load_mesh_uv(str(raw_path))
    DiscreteShellFrictionHardeningPlasticBending().apply_to(
        mesh, bending_stiffness=BENDING_STIFFNESS, hardening_k=Hardening_K,
        yield_theta=YIELD_THETA, plastic_theta=PLASTIC_THETA,
        friction_k=BENDING_FRICTION, thres_theta=THRES_THETA, anchor_theta=ANCHOR_THETA,
    )
    SoftPositionConstraint().apply_to(mesh, 200000.0)
    if not args.no_wind:
        try:
            from uipc.constitution import TriangleWind
        except ImportError as exc:
            raise RuntimeError("This scene requires a libuipc build with TriangleWind. "
                               "Run with --no-wind to preview folding and launch without wind.") from exc
        TriangleWind().apply_to(mesh, velocity=np.array(args.wind), enabled=False)
    obj = scene.objects().create("Paper plane")
    slot, _ = obj.geometries().create(mesh)
    engine = Engine("cuda", str(out / "backend"))
    world = World(engine)

    phase_one_end = rest.copy()
    launch_start = rest.copy()
    crease_target = rest.copy()
    launch_center = rest.mean(axis=0)

    def animate(info):
        geo = info.geo_slots()[0].geometry()
        frame = world.frame()
        constraints = view(geo.vertices().find(builtin.is_constrained))
        constraints[:] = int(frame < RELEASE)
        if frame >= RELEASE:
            return
        aim = view(geo.vertices().find(builtin.aim_position)).reshape(-1, 3)
        if frame <= CREASE_FRAMES:
            crease_angle = frame * np.pi / 100
            crease_target[:, 0] = rest[:, 0] * np.cos(crease_angle)
            crease_target[:, 1] = rest[:, 1] + np.abs(rest[:, 0]) * np.sin(crease_angle)
            aim[:] = crease_target
        elif frame <= FOLD_END:
            alpha = (frame - CREASE_FRAMES) / SHAPE_FRAMES
            aim[:] = (1 - alpha) * phase_one_end + alpha * target
        else:
            alpha = (frame - FOLD_END) / THROW_FRAMES
            pitch = alpha * angle
            c, s = np.cos(pitch), np.sin(pitch)
            rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
            aim[:] = (
                (launch_start - launch_center) @ rotation.T + launch_center
                + (frame - FOLD_END) * DT * launch_velocity
            )

    scene.animator().insert(obj, animate)
    world.init(scene)
    geo = slot.geometry()
    io.write(str(out / "cloth_0000.obj"), geo)

    def advance():
        next_frame = world.frame() + 1
        if not args.no_wind:
            view(geo.triangles().find("wind/enabled"))[:] = int(next_frame >= RELEASE)
        world.advance()
        world.retrieve()
        if world.frame() != next_frame:
            raise RuntimeError(f"Simulation did not reach frame {next_frame}.")
        positions = np.asarray(geo.positions().view()).reshape(-1, 3)
        if not np.isfinite(positions).all():
            raise RuntimeError(f"Nonfinite positions at frame {next_frame}.")
        if next_frame == CREASE_FRAMES:
            phase_one_end[:] = positions
        if next_frame == FOLD_END:
            launch_start[:] = positions
            launch_center[:] = positions.mean(axis=0)
        io.write(str(out / f"cloth_{next_frame:04d}.obj"), geo)
        if next_frame in (CREASE_FRAMES, FOLD_END, RELEASE, args.frames):
            print(f"Frame {next_frame}: center {positions.mean(axis=0)}", flush=True)

    if args.headless:
        for _ in range(args.frames):
            advance()
        return

    import polyscope as ps
    from polyscope import imgui

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    plane = ps.register_surface_mesh("Paper plane", rest, faces)
    playing = False

    def update():
        nonlocal playing
        frame = world.frame()
        stage = ("Folding" if frame < FOLD_END else
                 "Launching" if frame < RELEASE else "Flying")
        imgui.TextUnformatted(f"{stage} | Frame {frame}/{args.frames}")
        if imgui.Button("Pause" if playing else "Play"):
            playing = not playing
        imgui.SameLine()
        step = imgui.Button("One step")
        if frame < args.frames and (playing or step):
            advance()
            plane.update_vertex_positions(np.asarray(geo.positions().view()).reshape(-1, 3))
        if world.frame() >= args.frames:
            playing = False

    ps.set_user_callback(update)
    ps.show()


if __name__ == "__main__":
    main()
