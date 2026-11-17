from pathlib import Path
import ast,hashlib,json,os,sys,time,traceback
import numpy as np
R=Path(__file__).resolve().parent
case=json.loads((R/"case.json").read_text())
active=case["active_count"]
epochs=case["epochs"]
end_frame=40
state=dict(status='initializing',pid=os.getpid())
def save():
    t=R/'status.tmp.json';t.write_text(json.dumps(state,indent=2));t.replace(R/'status.json')
def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
save()
try:
    import torch
    torch.set_num_threads(4)
    source=(R/'original_snapshot.py').read_text(encoding='utf-8-sig').replace('\r\n','\n')
    source=source[:source.index('\nworld.dump()\n')]
    source=replace(source,'from diff_sim_friction_plastic_gated_module import *','from runtime_diff_module_copy import *\nfrom yarn_rotation_reference import effective_coefficients, return_mapping')
    source=replace(source,'from uipc.constitution import DiscreteShellFrictionHardeningPlasticBending, StVKNonlinearFrictionHardeningPlasticShell',
        'from uipc.constitution import DiscreteShellFrictionHardeningPlasticBending, StVKNonlinearYarnRotationFrictionHardeningPlasticShell')
    source=replace(source,'nlstckfhps = StVKNonlinearFrictionHardeningPlasticShell()',
        'nlstckfhps = StVKNonlinearYarnRotationFrictionHardeningPlasticShell()')
    source=replace(source,'mass_density=MASS_DENSITY, thickness=THICKNESS)',
        'mass_density=MASS_DENSITY, thickness=THICKNESS, enable_yarn_rotation=True)')
    source=replace(source,'scene = Scene(config)',
        'config["diff_sim"]["yarn_rotation_shear_history_gradient"] = False\n'
        'config["newton"]["velocity_tol"] = .05\nconfig["linear_system"]["tol_rate"] = .001\nscene = Scene(config)')
    source=replace(source,'_log_dir = os.path.join(os.path.dirname(__file__), "logs")','_log_dir = '+repr(str(R/'logs')))
    source=replace(source,'workspace = AssetDir.output_path(__file__)','workspace = '+repr(str(R/'backend')))
    # Replace data and mesh routing only; geometry, animation and material remain inherited.
    lines=source.splitlines(True)
    for node in reversed(ast.parse(source).body):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            n=node.targets[0].id
            if n in ['MESH_PATH','training_data_path','IS_SAVE_LOAD_DEFORMATION']:
                value={'MESH_PATH':str(R/'mesh.obj'),'training_data_path':str(R/'data.npy'),'IS_SAVE_LOAD_DEFORMATION':False}[n]
                lines[node.lineno-1:node.end_lineno]=[f'{n} = {value!r}\n']
    source=''.join(lines)
    import re
    source=replace(source,'tested_fabric_type = _cli_args.fabric','tested_fabric_type = '+repr(case['fabric']))
    material=dict(case['material']);material.update(dict(zip(['k33_0','k33_1','friction_k33','thres_uv','yield_uv','hardening_k33'],case['initial'])))
    source=replace(source,'_material = FABRIC_INITIAL_GUESSES[tested_fabric_type]','_material = '+repr(material))
    source=replace(source,'EndFrame = _cli_args.forward_frames - 1 if FORWARD_VALIDATION else 40','EndFrame = '+str(end_frame))
    assert active == 5 and end_frame == 40
    source=replace(source,'enumerate(PARAM_NAMES):','enumerate((*PARAM_NAMES[:4], PARAM_NAMES[5])):')
    source=replace(source,'class MyDiffSimParameter(DiffSimParameter):','SHEAR_PARAMETER_BOUNDS = (*SHEAR_PARAMETER_BOUNDS[:4], SHEAR_PARAMETER_BOUNDS[5])\nclass MyDiffSimParameter(DiffSimParameter):')
    source=replace(source,'parm = MyDiffSimParameter(scene=scene, size=6,','parm = MyDiffSimParameter(scene=scene, size=5,')
    source=replace(source,'p_ref=[K33_0, K33_1, Fric_K33, Thres_uv, Yield_UV, Hardening_K33])','p_ref=[K33_0, K33_1, Fric_K33, Thres_uv, Hardening_K33])')
    source=replace(source,'    P = parm.forward()\n','    active_P = parm.forward()\n    P = torch.cat((active_P[:4], active_P.new_tensor([Yield_UV]), active_P[4:]))\n')
    source,count=re.subn(r'^diff_yield_uv:[^\n]*\nview\(diff_yield_uv\)\[:\] = [^\n]*\n','',source,flags=re.M)
    assert count == 1
    source=replace(source,'view(diff_hardening_k33)[:] = 5','view(diff_hardening_k33)[:] = 4')

    start=source.index('    if is_rotate_yarns:\n',source.index('def net_force('))
    end=source.index('    NetForce_xy =',start)
    # Partial dE/dF keeps the constitutive state fixed during force evaluation.
    # Outer autograd still differentiates that force through parameter-dependent return mapping.
    new='''    q = G[1]
    parameters = torch.stack([torch.ones_like(q)*v for v in
        (K11,K22,K12,K33_0,K33_1,Friction_K33,thres_uvs.reshape(-1),yield_uv_ori.reshape(-1),hardening_k33)],dim=-1)
    strains = torch.stack((G[0],G[3],q),dim=-1)
    plastic_now,anchor_now,_,_ = return_mapping(strains,parameters,
        plastic_strain_uvs.reshape(-1),anchor_strain_uvs.reshape(-1),hardening_plastic_uvs.reshape(-1),True)
    def membrane_energy_fixed_state(F,parameters,plastic,anchor):
        C=F.transpose(-1,-2)@F
        strain=torch.stack((.5*(C[:,0,0]-1),.5*(C[:,1,1]-1),.5*C[:,0,1]),dim=-1)
        a,b,c,k=effective_coefficients(strain,parameters,True)
        u,v,s=strain.unbind(-1)
        return (.5*a*u.square()+c*u*v+.5*b*v.square()+k*(s-plastic).square()
            +parameters[:,5]*(s-anchor).square()).sum()
    derivative=torch.func.grad(membrane_energy_fixed_state,argnums=0)(F_s,parameters,plastic_now,anchor_now)
    local_gradient=(torch.bmm(Dut,derivative[:,:,0].unsqueeze(-1))
        +torch.bmm(Dvt,derivative[:,:,1].unsqueeze(-1))).squeeze(-1)
    Stretching_Force = THICKNESS * torch.mv(cloth_SF_T,(-cloth_faces_m.unsqueeze(-1)*local_gradient).reshape(-1))

'''
    source=source[:start]+new+source[end:]
    (R/'executed_training_source.py').write_text(source,encoding='utf-8')
    sys.argv=[str(R/'original_snapshot.py'),'--fabric','61_Wool','--epochs',str(epochs),'--headless-train']
    env=dict(__name__='__cotton_new_shear_training_copy__',__file__=str(R/'original_snapshot.py'))
    exec(compile(source,str(R/'executed_training_source.py'),'exec'),env)
    np.testing.assert_allclose(env['parm'].forward().detach().numpy(),[case['initial'][i] for i in (0, 1, 2, 3, 5)],rtol=1e-12)
    assert env['NUM_VERTICES']==1792 and env['NUM_FACES']==3456
    assert env['parm'].U().numel()==active
    assert env['EndFrame']==end_frame and env['PARAM_LR']==.02 and env['config']['contact']['enable'] is False
    env['world'].dump()
    captured={};best=float('inf');best_u=None;reference=float('inf');stale=0;rolling=[]
    def capture(loss,P,torques):
        captured.update(loss=float(loss.detach()),parameters=P.detach().cpu().numpy().copy(),prediction=torques.detach().cpu().numpy().copy())
    env['save_best_checkpoint']=capture
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='loss_fn')
    forward=ast.get_source_segment(source,fn).replace('def loss_fn(', 'def eval_loss(',1).replace('    loss.backward()','    # forward only').replace('    check_dead_gradients()','    # no gradient diagnostics in replay')
    exec(compile(forward,'<replay>','exec'),env)
    state.update(status='training',history_gradient=False,initial_parameters=env['parm'].forward().detach().numpy().tolist());save()
    assert env['Yield_UV'] == .002
    state.update(fixed_yield_uv=.002, frames=[0,40]);save()
    for epoch in range(epochs):
        if (R/'STOP').exists():state['stop_reason']='STOP requested';break
        t=time.perf_counter();u=env['parm'].U().detach().clone();env['_current_epoch']=epoch
        loss=float(env['optimizer'].step(env['loss_fn']).detach())
        gradient=env['parm'].U().grad.detach().cpu().numpy().copy()
        rolling.append(loss)
        if len(rolling)>=5:
            median=float(np.median(rolling[-5:]))
            if median<reference-max(1e-12,abs(reference)*.001) if np.isfinite(reference) else True: reference=median;stale=0
            else: stale+=1
        assert np.isfinite(loss) and np.isfinite(gradient).all()
        assert gradient.shape == (5,) and captured['parameters'][4] == .002
        if loss<best:
            best=loss;best_u=u.clone()
            np.savez(R/'best.npz',epoch=epoch,loss=loss,U=u.numpy(),parameters=captured['parameters'],prediction=captured['prediction'],gt=env['torque_gt'].numpy(),angle_deg=env['raw_twist_angles'][:end_frame+1])
            state.update(best_epoch=epoch,best_parameters=captured['parameters'].tolist())
        row=dict(epoch=epoch,loss=loss,best_loss=best,parameters=captured['parameters'].tolist(),gradient_U=gradient.tolist(),seconds=time.perf_counter()-t)
        with (R/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        torch.save(dict(epoch=epoch,U=env['parm'].U().detach().clone(),optimizer=env['optimizer'].state_dict()),R/'last_checkpoint.pt')
        state.update(epoch=epoch,latest=row,best_loss=best);save();print('EPOCH_SUMMARY',json.dumps(row),flush=True)
        if epoch>=29 and stale>=20:state['stop_reason']='rolling median plateau';break
    with torch.no_grad():
        env['parm'].U().copy_(best_u);env['parm'].sync();replay=float(env['eval_loss']())
    np.savez(R/'best_replay.npz',loss=replay,parameters=captured['parameters'],prediction=captured['prediction'],gt=env['torque_gt'].numpy(),angle_deg=env['raw_twist_angles'][:end_frame+1])
    state.update(status='complete',best_replay_loss=replay);save()
except BaseException:
    state.update(status='failed',error=traceback.format_exc());save();raise
