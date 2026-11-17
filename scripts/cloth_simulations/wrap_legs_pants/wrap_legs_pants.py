"""Independent pants wrapping GUI with five learned orthotropic fabrics.

Physical warp follows UV v by default, matching the original Up_Direction mapping.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mesh_and_motion import read_obj_uv, align_face_uv

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"


def wrap_offset(frame):
    """Lift by 0.25 m at frame 100, return at 200, release after 200."""
    return .25*max(0.,1.-abs(float(frame)-100.)/100.)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--fabric', choices=['cotton','linen','wool','pet','silk'], default='linen')
    cli.add_argument('--warp-axis', choices=['u','v'], default='v')
    cli.add_argument('--cloth', type=Path, default=ASSETS/'Pants_Higher_Res.obj')
    cli.add_argument('--body', type=Path, default=ASSETS/'body_bottom_part.obj')
    cli.add_argument('--frames', type=int, default=300)
    cli.add_argument('--headless-smoke', type=int, default=0)
    cli.add_argument('--check-only', action='store_true')
    cli.add_argument('--info-log', action='store_true', help='Print backend diagnostics during verification')
    shear_group = cli.add_mutually_exclusive_group()
    shear_group.add_argument('--enable-shear-history', action='store_true', default=True,
                             help='Use learned shear friction/plasticity thresholds (default)')
    shear_group.add_argument('--disable-shear-history', dest='enable_shear_history', action='store_false',
                             help='Suppress shear sliding/plasticity by setting both thresholds to 100')
    args = cli.parse_args()
    if args.frames < 1 or args.headless_smoke < 0 or args.headless_smoke > args.frames:
        cli.error('Require frames >= 1 and 0 <= headless-smoke <= frames')
    for path in (args.cloth,args.body):
        if not path.is_file():
            raise FileNotFoundError(path)
    x,faces,corners,edge_angles,audit = read_obj_uv(args.cloth)
    fixed_mask = (x[:,1] > .89).astype(np.int32)
    manipulated_mask = ((x[:,1] < .11) & (x[:,0] > 0.)).astype(np.int32)
    if not fixed_mask.any() or not manipulated_mask.any():
        raise ValueError('Original coordinate-based waist/hem selections are empty for this mesh')
    if np.any(fixed_mask & manipulated_mask):
        raise ValueError('Fixed and manipulated selections overlap')
    audit.update(fixed_vertices=int(fixed_mask.sum()),manipulated_vertices=int(manipulated_mask.sum()))
    record = json.loads((HERE/'learned_parameters.json').read_text(encoding='utf-8'))[args.fabric]
    p,m = record['bending'],record['stretch_shear']
    u,v = ('weft','warp') if args.warp_axis == 'v' else ('warp','weft')
    bending = dict(bending_stiffness_warp=p['kb_'+u],bending_stiffness_weft=p['kb_'+v],
                   bending_torsion=m['k33_0']*m['thickness']**3/12,bending_poisson=0.,
                   plastic_theta_warp=0.,plastic_theta_weft=0.,anchor_theta_warp=0.,anchor_theta_weft=0.)
    for backend,physical in [('friction_k','friction'),('hardening_k','hardening'),
                              ('yield_theta','yield'),('thres_theta','thres')]:
        bending[backend+'_warp'] = p[physical+'_'+u]
        bending[backend+'_weft'] = p[physical+'_'+v]
    print('MESH_AUDIT',json.dumps(audit),flush=True)
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
    (out/'backend').mkdir(parents=True)
    config = Scene.default_config()
    config['dt'] = .01
    config['contact']['enable'] = True
    config['contact']['d_hat'] = .0005
    config['contact']['friction']['enable'] = False
    config['newton']['velocity_tol'] = .05
    config['linear_system']['tol_rate'] = .001
    Logger.set_level(Logger.Level.Info if args.info_log else Logger.Level.Warn)
    engine = Engine('cuda',str(out/'backend'))
    world = World(engine)
    scene = Scene(config)
    scene.contact_tabular().default_model(0.,1e9)
    garment_elem = scene.contact_tabular().create('garment')
    body_elem = scene.contact_tabular().create('body_elem')
    scene.contact_tabular().insert(body_elem,body_elem,0.,0.,False)
    io = SimplicialComplexIO(Transform.Identity())
    mesh = io.read(str(args.cloth))
    label_surface(mesh)
    shear_threshold = m['thres'] if args.enable_shear_history else 100.
    shear_yield = m['yield_uv'] if args.enable_shear_history else 100.
    StVKNonlinearFrictionHardeningPlasticShell().apply_to(
        mesh,m[u],0.,m[v],m['k33_0'],m['k33_1'],
        hardening_k33=m['hardening'],friction_k33=m['friction'],
        thres_uv=shear_threshold,yield_uv=shear_yield,anchor_strain_uv=0.,plastic_strain_uv=0.,
        mass_density=m['density'],thickness=m['thickness'])
    DiscreteShellFrictionHardeningPlasticOrthotropicBending().apply_to(mesh,**bending)
    spc = SoftPositionConstraint()
    spc.apply_to(mesh,1000.)
    garment_elem.apply_to(mesh)
    cloth = scene.objects().create('cloth')
    slot,_ = cloth.geometries().create(mesh)
    geo = slot.geometry()
    backend_faces = np.asarray(geo.triangles().find(builtin.topo).view()).reshape(-1,3)
    np.testing.assert_allclose(geo.positions().view().reshape(-1,3),x,rtol=0.,atol=1e-12)
    view(geo.triangles().find('rest_x'))[:] = align_face_uv(faces,corners,backend_faces).reshape(-1,6,1)
    edges = np.asarray(geo.edges().find(builtin.topo).view()).reshape(-1,2)
    angles = np.array([edge_angles[tuple(sorted(map(int,e)))] for e in edges])
    angle_slot = geo.edges().find('bias_angle')
    view(angle_slot)[:] = angles.reshape(angle_slot.view().shape)
    initial = geo.positions().view().copy()
    fixed = view(geo.vertices().find(builtin.is_fixed))
    constrained = view(geo.vertices().find(builtin.is_constrained))
    fixed[:] = fixed_mask.reshape(fixed.shape)
    constrained[:] = manipulated_mask.reshape(constrained.shape)
    view(geo.vertices().find(builtin.aim_position))[:] = initial

    body = scene.objects().create('body')
    body_mesh = io.read(str(args.body))
    label_surface(body_mesh)
    Empty().apply_to(body_mesh,thickness=.0005)
    spc.apply_to(body_mesh,1000.)
    body_elem.apply_to(body_mesh)
    body_slot,_ = body.geometries().create(body_mesh)
    body_geo = body_slot.geometry()
    body_initial = body_geo.positions().view().copy()
    view(body_geo.vertices().find(builtin.is_constrained))[:] = 1
    view(body_geo.vertices().find(builtin.aim_position))[:] = body_initial

    def wrap_pants(info):
        frame = info.frame()
        constrained = view(geo.vertices().find(builtin.is_constrained))
        if frame > 200:
            constrained[:] = 0
            return
        constrained[:] = manipulated_mask.reshape(constrained.shape)
        target = initial.copy().reshape(-1,3)
        target[:,1] += wrap_offset(frame)
        view(geo.vertices().find(builtin.aim_position))[:] = target.reshape(initial.shape)

    scene.animator().insert(cloth,wrap_pants)
    print('WORLD_INIT_START',flush=True)
    world.init(scene)
    edge_names = ['dihedral_theta']+[name+'_'+d for name in
        ('plastic_theta','anchor_theta','yield_theta','hardening_plastic_theta') for d in ('warp','weft')]
    tri_names = ['anchor_strain_uv','plastic_strain_uv','yield_uv','hardening_plastic_uv']
    edge_initial = {n:geo.edges().find(n).view().copy() for n in edge_names}
    tri_initial = {n:geo.triangles().find(n).view().copy() for n in tri_names}
    if not world.dump():
        raise RuntimeError('Frame-0 dump failed')
    metadata = dict(fabric=args.fabric,warp_axis=args.warp_axis,material=record,backend_bending=bending,
                    shear_threshold=shear_threshold,shear_yield=shear_yield,config=config,
                    mesh=str(args.cloth),body=str(args.body),mesh_audit=audit,
                    cloth_constraint=1000.,body_constraint=1000.,contact_barrier_stiffness=1e9,
                    contact_friction=False,body_self_contact=False,
                    motion=dict(peak_frame=100,height_m=.25,return_frame=200,release_after_frame=200))
    (out/'configuration.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    sequence = {'number':0,'path':None}

    def recover():
        if not world.recover(0):
            raise RuntimeError('Frame-0 recovery failed')
        world.retrieve()
        if world.frame() != 0:
            raise RuntimeError('Recovery did not return to frame zero')
        for name,value in edge_initial.items():
            view(geo.edges().find(name))[:] = value
        for name,value in tri_initial.items():
            view(geo.triangles().find(name))[:] = value
        view(geo.edges().find('reset_anchor_plastic'))[:] = 1
        view(geo.triangles().find('reset_anchor_plastic'))[:] = 1
        view(geo.vertices().find(builtin.is_fixed))[:] = fixed_mask.reshape(fixed.shape)
        view(geo.vertices().find(builtin.is_constrained))[:] = manipulated_mask.reshape(constrained.shape)
        view(geo.vertices().find(builtin.aim_position))[:] = initial
        view(body_geo.vertices().find(builtin.is_constrained))[:] = 1
        view(body_geo.vertices().find(builtin.aim_position))[:] = body_initial
        sequence['number'] += 1
        sequence['path'] = out/f"sequence_{sequence['number']:03d}"
        sequence['path'].mkdir()
        io.write(str(sequence['path']/'cloth_0000.obj'),geo)

    def advance():
        tick = time.perf_counter()
        world.advance()
        world.retrieve()
        if world.frame() == 1:
            view(geo.edges().find('reset_anchor_plastic'))[:] = 0
            view(geo.triangles().find('reset_anchor_plastic'))[:] = 0
        if not np.isfinite(geo.positions().view()).all():
            raise RuntimeError('Nonfinite cloth positions')
        io.write(str(sequence['path']/f'cloth_{world.frame():04d}.obj'),geo)
        print('FRAME',world.frame(),'SECONDS',time.perf_counter()-tick,flush=True)

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


if __name__=='__main__':
    main()
