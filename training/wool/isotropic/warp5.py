import sys
sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parent))
"""Real Cotton warp five-parameter joint refinement with fixed best stiffness and friction, corrected isotropic readout.

Original model, historical scripts and GT are read-only. The verified synthetic
source snapshot supplies the force/model implementation; only material settings,
GT routing, frame window and the single active parameter are transformed.
"""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'template.py'
SNAPSHOT=ROOT/'source_snapshot.py'
GT=ROOT/'data.npy'
NAMES=['kb','friction','thres','hardening','yield']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--learning-rates',type=float,nargs=5,default=[.3,.3,.002,0.,0.])
    args=parser.parse_args()
    out=ROOT/'training'
    out.mkdir();(out/'backend').mkdir()
    status=dict(status='initializing',pid=os.getpid(),data=str(GT),active_parameters=NAMES)
    def save():
        (out/'status.tmp.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
        (out/'status.tmp.json').replace(out/'status.json')
    print('Output:',out,flush=True);save()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        # Read only the previously used material presets, never execute its model.
        material={'warp': 3368709.2254, 'weft': 2448119.559, 'density': 590, 'thickness': 0.0003, 'k33_0': 236633.21787943662, 'k33_1': 1370382.7490411499, 'friction': 310143.15690572525, 'thres': 100.0, 'yield_uv': 100.0, 'hardening': 1174590.6921811649};initial={'kb': 600.0, 'friction': 400.0, 'thres': 0.1, 'hardening': 50.0, 'yield': 100.0}
        initial.update(json.loads((ROOT.parent / 'warp3' / 'training' / 'status.json').read_text(encoding='utf-8'))['best_parameters'])
        status.update(initial_parameters=initial,learning_rates_U=args.learning_rates,initialization='pipeline stage initial parameters')
        hashes={str(p):digest(p) for p in [SOURCE,SNAPSHOT,GT,ROOT/'diff_sim_friction_plastic_gated_module.py',Path(str(ROOT / 'mesh.obj'))]}
        data=np.load(GT,allow_pickle=False)
        if data.ndim!=2 or data.shape[0]!=2 or data.shape[1]<=150 or not np.isfinite(data).all():raise ValueError('Invalid real GT')
        comp=float(data[0].max())/1000.
        if comp<=0:raise ValueError('Invalid compression displacement')
        np.testing.assert_allclose(data[0,:51]/1000.,np.arange(51)*comp/50.,atol=1e-8,rtol=1e-6)
        phase=np.arange(151)%100
        expected=np.minimum(phase,100-phase)*comp/50.
        np.testing.assert_allclose(data[0,:151]/1000.,expected,atol=1e-8,rtol=1e-6)
        selected_frames=np.r_[30:71,120:151]
        gt=data[1,selected_frames].copy()
        source=SNAPSHOT.read_text(encoding='utf-8-sig').replace('\r\n','\n')
        (out/'source_snapshot.py').write_text(source,encoding='utf-8')
        source=source[:source.index('if GUIInfo.enabled():')]
        values=dict(Warp_Stiffness=material['warp'],Weft_Stiffness=material['weft'],K33_0=material['k33_0'],K33_1=material['k33_1'],
                    Fric_K33=material['friction'],Hardening_K33=material['hardening'],MASS_DENSITY=material['density'],THICKNESS=material['thickness'],
                    Thres_uv=material['thres'],Yield_UV=material['yield_uv'],BENDING_STIFFNESS=initial['kb'],BENDING_FRICTION=initial['friction'],THRES_THETA=initial['thres'],
                    Hardening_K=initial['hardening'],YEILD_THETA=initial['yield'],tested_fabric_type='61_Wool',MESH_PATH=str(ROOT / 'mesh.obj'),Up_Direction='warp',EndFrame=150,TRAIN_START_FRAME=29,TRAIN_END_FRAME=150,
                    COMP_DIS=comp,PARAM_LR=1.0,NUM_EPOCH=args.epochs,workspace=str(out/'backend'))
        lines=source.splitlines(True);found=set()
        for node in reversed(ast.parse(source).body):
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in values:
                name=node.targets[0].id;found.add(name)
                lines[node.lineno-1:node.end_lineno]=[f'{name} = {values[name]!r}\n']
            elif isinstance(node,ast.With) and 'gt_comp_force = ' in ast.get_source_segment(source,node):
                lines[node.lineno-1:node.end_lineno]=['gt_comp_force = torch.tensor(_real_gt, dtype=torch.float64)\n']
        assert found==set(values),set(values)-found
        source=''.join(lines)
        assert source.count('for f in range(TRAIN_START_FRAME, TRAIN_END_FRAME):')==1
        source=source.replace('for f in range(TRAIN_START_FRAME, TRAIN_END_FRAME):', 'for f in '+repr((selected_frames-1).tolist())+':')
        marker="config['contact']['enable'] = False"
        assert source.count(marker)==1
        source=source.replace(marker,marker+"\nconfig['newton']['velocity_tol'] = 0.005\nconfig['linear_system']['tol_rate'] = 0.0001")
        def replace(old,new):
            nonlocal source
            assert source.count(old)==1,(old,source.count(old))
            source=source.replace(old,new,1)
        mapping=['bending_stiffness','friction_stiffness','thres_theta','hardening_k','yield_theta_ori']
        replace("diff_friction_k: AttributeSlotI32 = mesh.edges().create('diff/friction_stiffness', -1)\nview(diff_friction_k)[:] = 0",
                '\n'.join(f"joint_diff_{i} = mesh.edges().create('diff/{name}', -1)\nview(joint_diff_{i})[:] = {i}" for i,name in enumerate(mapping)))
        replace('parm = MyDiffSimParameter(scene=scene, size=1)','parm = MyDiffSimParameter(scene=scene, size=5)')
        replace('    u = np.sqrt(BENDING_FRICTION)   \n    parm.U().fill_(u)',
                '    parm.U().copy_(torch.sqrt(torch.tensor([BENDING_STIFFNESS, BENDING_FRICTION, THRES_THETA, Hardening_K, YEILD_THETA], dtype=torch.float64)))')
        replace('    parm_bending_friction_k_tensor = optimizer.param_groups[0]["params"][0]**2',
                '    joint_p = parm.forward().reshape(-1)\n    parm_bending_friction_k_tensor = joint_p[1]\n    thres_thetas_tensor = joint_p[2] * torch.ones_like(thres_thetas_tensor)\n    joint_yield = joint_p[4] * torch.ones_like(rest_yield_theta_ori_tensor)')
        replace('        force = net_force(X_pred, BENDING_STIFFNESS, parm_bending_friction_k_tensor, Hardening_K, rest_yield_theta_ori_tensor,',
                '        force = net_force(X_pred, joint_p[0], parm_bending_friction_k_tensor, joint_p[3], joint_yield,')
        # Scale Adam displacement per coordinate before the existing DiffSimLR hook.
        replace('optimizer = optim.Adam(diff_module.parameters(), lr=PARAM_LR)', '''optimizer = optim.Adam(diff_module.parameters(), lr=1.0)
joint_rates = torch.tensor('''+repr(args.learning_rates)+''', dtype=torch.float64)
joint_before = None
def joint_pre_step(optimizer, args, kwargs):
    global joint_before
    joint_before = parm.U().detach().clone()
def joint_post_step(optimizer, args, kwargs):
    with torch.no_grad():
        parm.U().copy_(joint_before + joint_rates * (parm.U() - joint_before))
optimizer.register_step_pre_hook(joint_pre_step)
optimizer.register_step_post_hook(joint_post_step)''')
        reference_code='\n# Initial normal strains from the same geometry and UV metric used by CUDA.\n# Leave Guv and its existing history/return-mapping convention unchanged.\ndef initial_normal_strains_copy():\n    with torch.no_grad():\n        initial_positions = torch.from_numpy(mesh.positions().view().copy().reshape(-1, 3))\n        initial_flat = initial_positions.reshape(-1)\n        initial_edge1 = torch.mv(cloth_face_indices_1, initial_flat)\n        initial_edge2 = torch.mv(cloth_face_indices_2, initial_flat)\n        initial_ds = torch.stack([initial_edge1, initial_edge2], 1).view(NUM_FACES, 3, 2)\n        initial_F = torch.bmm(initial_ds, cloth_faces_invDm)\n        initial_G = .5 * (torch.bmm(initial_F.transpose(1, 2), initial_F)\n                          - torch.eye(2, dtype=torch.float64).unsqueeze(0))\n        return initial_G[:,0,0].detach().clone(), initial_G[:,1,1].detach().clone()\n\nrest_strain_uus_tensor, rest_strain_vvs_tensor = initial_normal_strains_copy()\n'
        assert source.count('\ndef net_force(')==1
        source=source.replace('\ndef net_force(',reference_code+'\ndef net_force(')
        import re
        assert re.findall(r"^[^#\n]*\.create\('(diff/[^']+)'",source,re.M)==['diff/'+x for x in mapping]
        (out/'executed_source.py').write_text(source,encoding='utf-8')
        env={'__name__':'__correct_real_joint__','__file__':str(SOURCE),'_real_gt':gt.tolist()}
        exec(compile(source,str(out/'executed_source.py'),'exec'),env)
        assert env['Up_Direction']=='warp' and env['parm'].U().numel()==5
        assert env['NUM_VERTICES']==1820 and env['NUM_FACES']==3456
        protocol=dict(fabric='61_Wool',direction='warp',mesh=str(ROOT / 'mesh.obj'),normal_reference='initial geometric Guu/Gvv',data=str(GT),frames=selected_frames.tolist(),initial_bending=initial,material=material,shear_threshold=material['thres'],shear_yield=material['yield_uv'],
                      config=env['config'],learning_rates_U=args.learning_rates,parameterization='P=U^2',post_step_scaling=True,input_hashes=hashes,
                      data_shape=list(data.shape),compression_distance_m=comp,no_tare_shift=True,
                      early_stop='minimum 30 epochs; 20 without cumulative 0.1% improvement',max_epochs=args.epochs)
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        np.savez(out/'training_data.npz',frames=selected_frames,force=gt,displacement_mm=data[0,selected_frames])
        print('PROTOCOL',json.dumps(protocol),flush=True)
        forces=[];actual=env['net_force']
        def capture(*a,**kw):
            result=actual(*a,**kw);forces.append(float(result.detach().item()));return result
        env['net_force']=capture
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='loss_fn')
        forward=ast.get_source_segment(source,node).replace('def loss_fn(', 'def eval_loss(',1).replace('    loss.backward()','    # forward only')
        exec(compile(forward,'<forward-eval>','exec'),env)
        with torch.no_grad():
            env['lr']._old_U.copy_(env['parm'].U());env['lr']._old_P.copy_(env['parm'].forward())
        best=float('inf');reference=float('inf');best_u=None;stale=0;rows=[]
        status['status']='training';save()
        for epoch in range(args.epochs):
            tick=time.perf_counter();forces=[]
            u=env['parm'].U().detach().clone();p=env['parm'].forward().detach().numpy().copy()
            loss=float(env['optimizer'].step(env['loss_fn']).detach().item())
            grad=env['parm'].U().grad.detach().numpy().copy();after=env['parm'].forward().detach().numpy().copy()
            if not np.isfinite(loss) or not np.isfinite(grad).all() or not np.isfinite(after).all():raise RuntimeError('Nonfinite training state')
            if epoch==0:
                np.savez(out/'initial_baseline.npz',parameters=p,names=NAMES,U=u.numpy(),loss=loss,predicted_force=forces,gt_force=gt,frames=selected_frames)
            if loss<best:
                best,best_u=loss,u
                np.savez(out/'best.npz',parameters=p,names=NAMES,U=u.numpy(),loss=loss,epoch=epoch,predicted_force=forces,gt_force=gt,frames=selected_frames)
                status.update(best_epoch=epoch,best_parameters=dict(zip(NAMES,p.tolist())))
            if not np.isfinite(reference) or loss<reference-max(1e-12,abs(reference)*.001):reference=loss;stale=0
            else:stale+=1
            row=dict(epoch=epoch,parameters=p.tolist(),after_step=after.tolist(),loss=loss,gradient_U=grad.tolist(),seconds=time.perf_counter()-tick,best_loss=best,stale=stale)
            rows.append(row)
            with (out/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            torch.save(dict(U=env['parm'].U().detach().clone(),optimizer=env['optimizer'].state_dict(),old_U=env['lr']._old_U,old_P=env['lr']._old_P,epoch=epoch),out/'last_checkpoint.pt')
            status.update(latest=row,epoch=epoch,initial_loss=rows[0]['loss'],best_loss=best);save()
            print('EPOCH_SUMMARY',json.dumps(row),flush=True)
            if epoch>=29 and stale>=20:status['stop_reason']='20 epochs without 0.1% improvement';break
        else:status['stop_reason']='epoch limit'
        with torch.no_grad():
            env['parm'].U().copy_(best_u);env['parm'].sync();forces=[]
            replay=float(env['eval_loss']().item())
        np.savez(out/'best_replay.npz',parameters=[status['best_parameters'][k] for k in NAMES],names=NAMES,loss=replay,predicted_force=forces,gt_force=gt,frames=selected_frames)
        unchanged=all(digest(Path(p))==sha for p,sha in hashes.items())
        status.update(status='complete',best_replay_loss=replay,original_inputs_unchanged=unchanged,epochs=len(rows))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        axes[0].semilogy([r['loss'] for r in rows]);axes[0].set_title('Training MSE')
        axes[1].plot(np.array([r['parameters'] for r in rows])/np.array([initial[k] for k in NAMES]));axes[1].legend(NAMES);axes[1].set_title('Parameter / initial')
        axes[2].plot(selected_frames,gt,label='GT');axes[2].plot(selected_frames,forces,label='Best replay');axes[2].legend()
        fig.tight_layout();fig.savefig(out/'training.png',dpi=150);plt.close(fig)
        save();print('RESULT',json.dumps(status,indent=2),flush=True)
    except Exception:
        status.update(status='failed',error=traceback.format_exc());save();raise


if __name__=='__main__':
    assert sys.stdin.readline().strip() == 'START'
    main()
