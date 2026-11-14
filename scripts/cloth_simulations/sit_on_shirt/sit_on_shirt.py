"""Independent sitting-shirt GUI with the learned orthotropic material.

Default: Wool, with physical warp along UV v.  This UV orientation
is an explicit assumption; inspect the garment UV layout before interpreting it.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from mesh_and_motion import read_obj_uv, align_face_uv, body_offset, lift_fraction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = HERE.parent / "assets"
DEFAULT_CLOTH = ASSETS / "shirt_high_res_processed.obj"
DEFAULT_SEAT = ASSETS / "seat.obj"
DEFAULT_BODY = ASSETS / "sit_body.obj"
HANG_VERTICES = [9304,9301,9297,9294,9290,9286,9283,132,135,140,143,147,149,152,38540,38781,38446]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--fabric', choices=['cotton','linen','wool','pet','silk'], default='wool',
                     help='Learned orthotropic bending and stretching/shearing material preset')
    cli.add_argument('--warp-axis', choices=['u','v'], default='v')
    cli.add_argument('--cloth', type=Path, default=DEFAULT_CLOTH)
    cli.add_argument('--seat', type=Path, default=DEFAULT_SEAT)
    cli.add_argument('--body', type=Path, default=DEFAULT_BODY)
    cli.add_argument('--frames', type=int, default=500)
    cli.add_argument('--headless-smoke', type=int, default=0)
    cli.add_argument('--check-only', action='store_true')
    shear_group = cli.add_mutually_exclusive_group()
    shear_group.add_argument('--enable-shear-history', action='store_true', default=True,
                             help='Use learned shear friction/plasticity thresholds (default)')
    shear_group.add_argument('--disable-shear-history', dest='enable_shear_history', action='store_false',
                             help='Suppress shear sliding/plasticity by setting both thresholds to 100')
    args = cli.parse_args()
    for p in (args.cloth,args.seat,args.body):
        if not p.is_file():
            raise FileNotFoundError(p)
    x,faces,corners,edge_angles,audit = read_obj_uv(args.cloth)
    if max(HANG_VERTICES) >= len(x):
        raise ValueError('Hanging vertex IDs belong to the original high-resolution shirt; supply that mesh')
    parameter_record = json.loads((HERE/'learned_parameters.json').read_text())[args.fabric]
    p, m = parameter_record['bending'], parameter_record['stretch_shear']
    # Same physical-to-model mapping as training: model warp is UV-u and model
    # weft is UV-v.  V-warp therefore swaps every directional material quantity.
    u,v = ('weft','warp') if args.warp_axis == 'v' else ('warp','weft')
    bending = dict(bending_stiffness_warp=p['kb_'+u],bending_stiffness_weft=p['kb_'+v],
                   bending_torsion=m['k33_0']*m['thickness']**3/12,bending_poisson=0.,
                   friction_k_warp=p['friction_'+u],friction_k_weft=p['friction_'+v],
                   hardening_k_warp=p['hardening_'+u],hardening_k_weft=p['hardening_'+v],
                   yield_theta_warp=p['yield_'+u],yield_theta_weft=p['yield_'+v],
                   thres_theta_warp=p['thres_'+u],thres_theta_weft=p['thres_'+v],
                   plastic_theta_warp=0.,plastic_theta_weft=0.,anchor_theta_warp=0.,anchor_theta_weft=0.)
    assert body_offset(50,1.) == 0. and body_offset(100,1.) == -1.
    assert body_offset(120,1.) == -1. and body_offset(170,1.) == 0.
    assert lift_fraction(170) == 0. and lift_fraction(220) == 1.
    print('MESH_AUDIT',json.dumps(audit),flush=True)
    print('PARAMETER_SOURCE',parameter_record['source'],flush=True)
    print('PHYSICAL_BENDING',json.dumps(p),flush=True)
    print('BACKEND_BENDING',json.dumps(bending),flush=True)
    if audit['shared_edges_with_uv_direction_disagreement']:
        print('WARNING: adjacent UV islands disagree on some edge directions; first incident face is used.',flush=True)
    if args.check_only:
        print('CHECK_ONLY_PASSED',flush=True)
        return

    import polyscope as ps
    from polyscope import imgui
    from uipc import view,builtin,Logger,Transform
    from uipc.core import Engine,World,Scene
    from uipc.geometry import SimplicialComplexIO,label_surface
    from uipc.constitution import (Empty,SoftPositionConstraint,
        StVKNonlinearFrictionHardeningPlasticShell,
        DiscreteShellFrictionHardeningPlasticOrthotropicBending)
    from uipc.gui import SceneGUI,GUIInfo

    out = HERE/'runs'/(args.fabric+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    (out/'meshes').mkdir(parents=True)
    (out/'backend').mkdir()
    config = Scene.default_config()
    config['dt'] = .01
    config['contact']['enable'] = True
    config['contact']['d_hat'] = .001  # Original body/seat scene clearance.
    config['newton']['velocity_tol'] = .05
    config['linear_system']['tol_rate'] = .001
    config['newton']['max_iter'] = 1024
    config['line_search']['max_iter'] = 8
    config['newton']['use_adaptive_tol'] = False
    Logger.set_level(Logger.Level.Warn)
    engine = Engine('cuda',str(out/'backend'))
    world = World(engine)
    scene = Scene(config)
    io = SimplicialComplexIO(Transform.Identity())
    spc = SoftPositionConstraint()

    cloth_mesh = io.read(str(args.cloth))
    label_surface(cloth_mesh)
    material = StVKNonlinearFrictionHardeningPlasticShell()
    shear_threshold = m['thres'] if args.enable_shear_history else 100.
    shear_yield = m['yield_uv'] if args.enable_shear_history else 100.
    material.apply_to(cloth_mesh,m[u],0.,m[v],m['k33_0'],m['k33_1'],
                       hardening_k33=m['hardening'],friction_k33=m['friction'],
                       thres_uv=shear_threshold,anchor_strain_uv=0.,yield_uv=shear_yield,
                       plastic_strain_uv=0.,mass_density=m['density'],thickness=m['thickness'])
    DiscreteShellFrictionHardeningPlasticOrthotropicBending().apply_to(cloth_mesh,**bending)
    spc.apply_to(cloth_mesh,2000.)
    cloth = scene.objects().create('cloth')
    cloth_slot,_ = cloth.geometries().create(cloth_mesh)
    geo = cloth_slot.geometry()
    backend_faces = geo.triangles().find(builtin.topo).view()
    view(geo.triangles().find('rest_x'))[:] = align_face_uv(faces,corners,backend_faces).reshape(-1,6,1)
    backend_edges = np.asarray(geo.edges().find(builtin.topo).view()).reshape(-1,2)
    angles = np.array([edge_angles[tuple(sorted(map(int,e)))] for e in backend_edges])
    view(geo.edges().find('bias_angle'))[:] = angles.reshape(geo.edges().find('bias_angle').view().shape)
    cloth_initial = geo.positions().view().copy()

    seat = scene.objects().create('seat')
    seat_mesh = io.read(str(args.seat))
    label_surface(seat_mesh)
    Empty().apply_to(seat_mesh,thickness=.001)
    seat_slot,_ = seat.geometries().create(seat_mesh)
    view(seat_slot.geometry().vertices().find(builtin.is_fixed))[:] = 1
    body = scene.objects().create('body')
    body_mesh = io.read(str(args.body))
    label_surface(body_mesh)
    Empty().apply_to(body_mesh,thickness=.001)
    spc.apply_to(body_mesh,1000.)
    body_slot,_ = body.geometries().create(body_mesh)
    body_geo = body_slot.geometry()
    view(body_geo.vertices().find(builtin.is_constrained))[:] = 1
    body_initial = body_geo.positions().view().copy()
    hang = np.zeros(len(x),dtype=np.int32)
    hang[HANG_VERTICES] = 1
    lift_state = {'start':None}
    distance = (1.83151-.532573)-.04

    def sit_down(info):
        frame = info.frame()
        target = body_initial.copy().reshape(-1,3)
        target[:,1] += body_offset(frame,distance)
        view(body_geo.vertices().find(builtin.aim_position))[:] = target.reshape(body_initial.shape)

    def lift_cloth(info):
        frame = info.frame()
        constrained = view(geo.vertices().find(builtin.is_constrained))
        if frame < 170:
            constrained[:] = 0
            return
        constrained[:] = hang.reshape(constrained.shape)
        if lift_state['start'] is None:
            lift_state['start'] = geo.positions().view().copy()
        alpha = lift_fraction(frame)
        target = (1-alpha)*lift_state['start']+alpha*cloth_initial
        view(geo.vertices().find(builtin.aim_position))[:] = target

    scene.animator().insert(body,sit_down)
    scene.animator().insert(cloth,lift_cloth)
    world.init(scene)
    edge_names = ['dihedral_theta']+[name+'_'+d for name in
        ('plastic_theta','anchor_theta','yield_theta','hardening_plastic_theta') for d in ('warp','weft')]
    tri_names = ['anchor_strain_uv','plastic_strain_uv','yield_uv','hardening_plastic_uv']
    edge_initial = {n:geo.edges().find(n).view().copy() for n in edge_names}
    tri_initial = {n:geo.triangles().find(n).view().copy() for n in tri_names}
    if not world.dump():
        raise RuntimeError('Frame-0 dump failed')
    metadata = dict(fabric=args.fabric,warp_axis=args.warp_axis,mesh=str(args.cloth),mesh_audit=audit,
                    material=parameter_record,backend_bending=bending,shear_threshold=shear_threshold,
                    shear_yield=shear_yield,config=config,cloth_constraint=2000,body_constraint=1000)
    (out/'configuration.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)

    def recover():
        if not world.recover(0):
            raise RuntimeError('Frame-0 recovery failed')
        world.retrieve()
        lift_state['start'] = None
        for name,value in edge_initial.items():
            view(geo.edges().find(name))[:] = value
        for name,value in tri_initial.items():
            view(geo.triangles().find(name))[:] = value
        view(geo.edges().find('reset_anchor_plastic'))[:] = 1
        view(geo.triangles().find('reset_anchor_plastic'))[:] = 1

    def advance():
        world.advance()
        world.retrieve()
        if world.frame() == 1:
            view(geo.edges().find('reset_anchor_plastic'))[:] = 0
            view(geo.triangles().find('reset_anchor_plastic'))[:] = 0
        io.write(str(out/'meshes'/f'cloth_{world.frame():04d}.obj'),geo)
        print('FRAME',world.frame(),flush=True)

    recover()
    if args.headless_smoke:
        for _ in range(args.headless_smoke):
            advance()
        print('SMOKE_PASSED',world.frame(),flush=True)
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
