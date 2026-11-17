"""Independent washing GUI using five learned orthotropic fabric presets.

Physical warp follows UV v by default, matching the original direction mapping.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from mesh_and_motion import read_obj_uv, align_face_uv

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
FIX_VERTICES = [2735,2736,2737,2738,2739,2740,2741,2743,2742,2744,2745,2746,
                2747,2748,2749,2750,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]


def chamber_angle(frame):
    """Absolute rotation: reverse every 50 frames, stop at frame 350."""
    elapsed = float(np.clip(frame - 100, 0, 250))
    return .03 * sum(sign * np.clip(elapsed - start, 0, 50)
                     for start, sign in ((0,1),(50,-1),(100,1),(150,-1),(200,1)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--fabric', choices=['cotton','linen','wool','pet','silk'], default='linen')
    cli.add_argument('--warp-axis', choices=['u','v'], default='v')
    cli.add_argument('--cloth', type=Path, default=ASSETS/'vest_high_res.obj')
    cli.add_argument('--chamber', type=Path, default=ASSETS/'wm_chamber.obj')
    cli.add_argument('--frames', type=int, default=500)
    cli.add_argument('--headless-smoke', type=int, default=0)
    cli.add_argument('--check-only', action='store_true')
    shear_group = cli.add_mutually_exclusive_group()
    shear_group.add_argument('--enable-shear-history', action='store_true', default=True,
                             help='Use learned shear friction/plasticity thresholds (default)')
    shear_group.add_argument('--disable-shear-history', dest='enable_shear_history', action='store_false',
                             help='Suppress shear sliding/plasticity by setting both thresholds to 100')
    args = cli.parse_args()
    if args.frames < 1 or args.headless_smoke < 0 or args.headless_smoke > args.frames:
        cli.error('Require frames >= 1 and 0 <= headless-smoke <= frames')
    for path in (args.cloth, args.chamber):
        if not path.is_file():
            raise FileNotFoundError(path)
    x, faces, corners, edge_angles, audit = read_obj_uv(args.cloth)
    if max(FIX_VERTICES) >= len(x):
        raise ValueError('Lifting vertex IDs require the original high-resolution vest mesh')
    record = json.loads((HERE/'learned_parameters.json').read_text(encoding='utf-8'))[args.fabric]
    p, m = record['bending'], record['stretch_shear']
    u, v = ('weft','warp') if args.warp_axis == 'v' else ('warp','weft')
    bending = dict(bending_stiffness_warp=p['kb_'+u], bending_stiffness_weft=p['kb_'+v],
                   bending_torsion=m['k33_0']*m['thickness']**3/12, bending_poisson=0.,
                   plastic_theta_warp=0., plastic_theta_weft=0.,
                   anchor_theta_warp=0., anchor_theta_weft=0.)
    for backend, physical in [('friction_k','friction'),('hardening_k','hardening'),
                               ('yield_theta','yield'),('thres_theta','thres')]:
        bending[backend+'_warp'] = p[physical+'_'+u]
        bending[backend+'_weft'] = p[physical+'_'+v]
    print('MESH_AUDIT', json.dumps(audit), flush=True)
    print('PHYSICAL_BENDING', json.dumps(p), flush=True)
    print('BACKEND_BENDING', json.dumps(bending), flush=True)
    if audit['shared_edges_with_uv_direction_disagreement']:
        print('WARNING: adjacent UV islands disagree on some edge directions; first incident face is used.', flush=True)
    if args.check_only:
        print('CHECK_ONLY_PASSED', flush=True)
        return

    import polyscope as ps
    from polyscope import imgui
    from uipc import view, builtin, Logger, Transform
    from uipc.core import Engine, World, Scene
    from uipc.geometry import SimplicialComplexIO, label_surface
    from uipc.constitution import (Empty, SoftPositionConstraint,
        StVKNonlinearFrictionHardeningPlasticShell,
        DiscreteShellFrictionHardeningPlasticOrthotropicBending)
    from uipc.gui import SceneGUI, GUIInfo

    out = HERE/'runs'/(args.fabric+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    (out/'backend').mkdir(parents=True)
    config = Scene.default_config()
    config['dt'] = .01
    config['contact']['enable'] = True
    config['contact']['d_hat'] = .001
    config['newton']['velocity_tol'] = .05
    config['linear_system']['tol_rate'] = .001
    Logger.set_level(Logger.Level.Warn)
    engine = Engine('cuda', str(out/'backend'))
    world = World(engine)
    scene = Scene(config)
    io = SimplicialComplexIO(Transform.Identity())
    mesh = io.read(str(args.cloth))
    label_surface(mesh)
    shear_threshold = m['thres'] if args.enable_shear_history else 100.
    shear_yield = m['yield_uv'] if args.enable_shear_history else 100.
    StVKNonlinearFrictionHardeningPlasticShell().apply_to(
        mesh, m[u], 0., m[v], m['k33_0'], m['k33_1'],
        hardening_k33=m['hardening'], friction_k33=m['friction'],
        thres_uv=shear_threshold, yield_uv=shear_yield,
        anchor_strain_uv=0., plastic_strain_uv=0.,
        mass_density=m['density'], thickness=m['thickness'])
    DiscreteShellFrictionHardeningPlasticOrthotropicBending().apply_to(mesh, **bending)
    spc = SoftPositionConstraint()
    spc.apply_to(mesh, 2000.)
    cloth = scene.objects().create('cloth')
    slot, _ = cloth.geometries().create(mesh)
    geo = slot.geometry()
    view(geo.triangles().find('rest_x'))[:] = align_face_uv(
        faces, corners, geo.triangles().find(builtin.topo).view()).reshape(-1,6,1)
    edges = np.asarray(geo.edges().find(builtin.topo).view()).reshape(-1,2)
    angles = np.array([edge_angles[tuple(sorted(map(int,e)))] for e in edges])
    angle_slot = geo.edges().find('bias_angle')
    view(angle_slot)[:] = angles.reshape(angle_slot.view().shape)
    initial = geo.positions().view().copy()
    hang = np.zeros(len(x), dtype=np.int32)
    hang[FIX_VERTICES] = 1

    chamber = scene.objects().create('washing_machine')
    chamber_mesh = io.read(str(args.chamber))
    label_surface(chamber_mesh)
    Empty().apply_to(chamber_mesh, thickness=.001)
    spc.apply_to(chamber_mesh, 1000.)
    chamber_slot, _ = chamber.geometries().create(chamber_mesh)
    chamber_geo = chamber_slot.geometry()
    view(chamber_geo.vertices().find(builtin.is_constrained))[:] = 1
    chamber_initial = chamber_geo.positions().view().copy()
    lift = {'start': None}

    def rotate_chamber(info):
        theta = chamber_angle(info.frame())
        c, s = np.cos(theta), np.sin(theta)
        rotation = np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
        target = chamber_initial.reshape(-1,3) @ rotation.T
        view(chamber_geo.vertices().find(builtin.aim_position))[:] = target.reshape(chamber_initial.shape)

    def lift_cloth(info):
        frame = info.frame()
        constrained = view(geo.vertices().find(builtin.is_constrained))
        if frame < 350:
            constrained[:] = 0
            return
        constrained[:] = hang.reshape(constrained.shape)
        if lift['start'] is None:
            lift['start'] = geo.positions().view().copy()
        alpha = float(np.clip((frame-350)/100., 0., 1.))
        view(geo.vertices().find(builtin.aim_position))[:] = (1-alpha)*lift['start']+alpha*initial

    scene.animator().insert(chamber, rotate_chamber)
    scene.animator().insert(cloth, lift_cloth)
    world.init(scene)
    edge_names = ['dihedral_theta']+[name+'_'+d for name in
        ('plastic_theta','anchor_theta','yield_theta','hardening_plastic_theta') for d in ('warp','weft')]
    tri_names = ['anchor_strain_uv','plastic_strain_uv','yield_uv','hardening_plastic_uv']
    edge_initial = {n:geo.edges().find(n).view().copy() for n in edge_names}
    tri_initial = {n:geo.triangles().find(n).view().copy() for n in tri_names}
    if not world.dump():
        raise RuntimeError('Frame-0 dump failed')
    metadata = dict(fabric=args.fabric, warp_axis=args.warp_axis, material=record,
                    backend_bending=bending, shear_threshold=shear_threshold, shear_yield=shear_yield,
                    config=config, mesh=str(args.cloth), chamber=str(args.chamber), mesh_audit=audit,
                    cloth_constraint=2000., chamber_constraint=1000.)
    (out/'configuration.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('OUTPUT', out, flush=True)
    sequence = {'number':0, 'path':None}

    def recover():
        if not world.recover(0):
            raise RuntimeError('Frame-0 recovery failed')
        world.retrieve()
        lift['start'] = None
        for name, value in edge_initial.items():
            view(geo.edges().find(name))[:] = value
        for name, value in tri_initial.items():
            view(geo.triangles().find(name))[:] = value
        view(geo.edges().find('reset_anchor_plastic'))[:] = 1
        view(geo.triangles().find('reset_anchor_plastic'))[:] = 1
        sequence['number'] += 1
        sequence['path'] = out/f"sequence_{sequence['number']:03d}"
        sequence['path'].mkdir()
        io.write(str(sequence['path']/'cloth_0000.obj'), geo)

    def advance():
        world.advance()
        world.retrieve()
        if world.frame() == 1:
            view(geo.edges().find('reset_anchor_plastic'))[:] = 0
            view(geo.triangles().find('reset_anchor_plastic'))[:] = 0
        io.write(str(sequence['path']/f'cloth_{world.frame():04d}.obj'), geo)
        print('FRAME', world.frame(), flush=True)

    recover()
    if args.headless_smoke:
        for _ in range(args.headless_smoke):
            advance()
        print('SMOKE_PASSED', world.frame(), flush=True)
    elif GUIInfo.enabled():
        ps.init()
        ps.set_ground_plane_mode('none')
        gui = SceneGUI(scene)
        gui.register()
        running = False

        def update():
            nonlocal running
            imgui.Text(f'Frame: {world.frame()} | Fabric: {args.fabric} | Warp UV axis: {args.warp_axis}')
            if imgui.Button('Run / Pause'):
                running = not running
            if imgui.Button('Reset'):
                running = False
                recover()
                gui.update()
            if imgui.Button('One Step'):
                running = False
                if world.frame() < args.frames:
                    advance()
                    gui.update()
            if running:
                if world.frame() < args.frames:
                    advance()
                    gui.update()
                else:
                    running = False

        ps.set_user_callback(update)
        ps.show()


if __name__ == '__main__':
    main()
