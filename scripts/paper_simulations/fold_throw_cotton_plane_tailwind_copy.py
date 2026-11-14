"""Fold the plane mesh with fitted Cotton, launch it, and apply an independent wind load."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from uipc import Logger, builtin, view
from uipc.constitution import (
    DiscreteShellFrictionHardeningPlasticOrthotropicBending,
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

# Cotton stretch, shear, and orthotropic bending preset from 2026-11-05.
MASS_DENSITY = 430.0
THICKNESS = 0.0004
Warp_Stiffness = 5753754.3771
Weft_Stiffness = 1229799.9644
K11 = Warp_Stiffness
K22 = Weft_Stiffness
K12 = 0.0

K33_0 = 418811.2270184079
K33_1 = 449440.2107390804
Fric_K33 = 607111.0711703744
Hardening_K33 = 108828.13543624771
Thres_uv = 0.0053157712329678294
Yield_UV = 0.022090133241914264

BENDING_STIFFNESS_WARP = 488.98048444696354
BENDING_STIFFNESS_WEFT = 187.80616871022428
BENDING_FRICTION_WARP = 540.7861057052156
BENDING_FRICTION_WEFT = 155.82203096291212
THRES_THETA_WARP = 0.09817093293022802
THRES_THETA_WEFT = 0.0626184467977323
Hardening_K_WARP = 52.04134477512193
Hardening_K_WEFT = 21.865392737541345
YIELD_THETA_WARP = 1.0080765163732834
YIELD_THETA_WEFT = 0.9522375149540244
BENDING_TORSION = K33_0 * THICKNESS**3 / 12.0
BENDING_POISSON = 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--wind", type=float, nargs=3, default=(0.0, 4.0, -10.0),
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
           datetime.now().strftime("fold_throw_cotton_tailwind_%Y%m%d_%H%M%S_%f"))
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
    face_uv = load_mesh_uv(str(raw_path)).reshape(-1, 3, 2)
    view(mesh.triangles().find("rest_x")).reshape(-1, 6)[:] = face_uv.reshape(-1, 6)
    DiscreteShellFrictionHardeningPlasticOrthotropicBending().apply_to(
        mesh, bending_stiffness_warp=BENDING_STIFFNESS_WARP,
        bending_stiffness_weft=BENDING_STIFFNESS_WEFT,
        bending_torsion=BENDING_TORSION, bending_poisson=BENDING_POISSON,
        hardening_k_warp=Hardening_K_WARP, hardening_k_weft=Hardening_K_WEFT,
        friction_k_warp=BENDING_FRICTION_WARP, friction_k_weft=BENDING_FRICTION_WEFT,
        thres_theta_warp=THRES_THETA_WARP, thres_theta_weft=THRES_THETA_WEFT,
        yield_theta_warp=YIELD_THETA_WARP, yield_theta_weft=YIELD_THETA_WEFT,
        plastic_theta_warp=0.0, plastic_theta_weft=0.0,
        anchor_theta_warp=0.0, anchor_theta_weft=0.0,
    )
    # Use the same material convention as the Cotton preset: UV U is warp.
    edge_angles = {}
    for face, uv in zip(faces, face_uv):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            du, dv = uv[b] - uv[a]
            edge = tuple(sorted((face[a], face[b])))
            angle = np.arctan2(abs(dv), abs(du))
            if not np.isclose(edge_angles.setdefault(edge, angle), angle, atol=1e-6):
                raise ValueError(f"Inconsistent material direction on edge {edge}.")
    edges = mesh.edges().topo().view().reshape(-1, 2)
    view(mesh.edges().find("bias_angle")).reshape(-1)[:] = [
        edge_angles[tuple(sorted(edge))] for edge in edges
    ]
    SoftPositionConstraint().apply_to(mesh, 200000.0)
    if not args.no_wind:
        try:
            from uipc.constitution import TriangleWind
        except ImportError as exc:
            raise RuntimeError("This scene requires a libuipc build with TriangleWind. "
                               "Run with --no-wind to preview folding and launch without wind.") from exc
        TriangleWind().apply_to(mesh, velocity=np.array(args.wind), enabled=False)
    obj = scene.objects().create("Cotton plane")
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
    plane = ps.register_surface_mesh("Cotton plane", rest, faces)
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
