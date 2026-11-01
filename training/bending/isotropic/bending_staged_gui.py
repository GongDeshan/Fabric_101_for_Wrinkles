"""Three-stage isotropic compression GUI, using the verified force implementation."""
import argparse,ast,contextlib,datetime,gc,hashlib,io,json,math
from pathlib import Path
import time,traceback
import numpy as np
import torch
import polyscope as ps
from polyscope import imgui
from stretch_logging_theme import set_logging_theme
from stretch_force_plot import draw_force_overlay

NAMES=('kb','friction','thres','hardening','yield')
ATTRS=('bending_stiffness','friction_stiffness','thres_theta','hardening_k','yield_theta_ori')
def active_count(settings,stage):return 3 if stage==1 or not settings['plastic'] else 5
def frames(stage):
    return list(range(30,71)) if stage==1 else list(range(30,71))+list(range(120,151))
def direction(stage):return 'weft' if stage==3 else 'warp'
def validate(settings):
    if type(settings.get('plastic')) is not bool:raise ValueError('Explicit boolean bending plasticity setting required')
    for axis in ('warp','weft'):
        data=np.load(settings[axis+'_data'],allow_pickle=False)
        if data.ndim!=2 or data.shape[0]!=2 or data.shape[1]<151 or not np.isfinite(data).all():
            raise ValueError(axis+': expected finite (2,N) compression data with N >=151')
        peak=float(data[0].max())
        phase=np.arange(151)%100
        expected=np.minimum(phase,100-phase)*peak/50
        if peak<=0 or not np.allclose(data[0,:151],expected,atol=1e-5,rtol=1e-5):
            raise ValueError(axis+': first 151 displacements must match a 100-step loading/unloading cycle, starting at zero. Units: mm.')
    values=settings['parameters']+settings['rates']+[settings['density'],settings['thickness'],settings['warp'],settings['weft'],settings['newton'],settings['linear']]
    if not np.isfinite(values).all() or min(values)<=0:raise ValueError('Parameters, rates and physical settings must be finite and positive.')
    if min(settings['epochs'])<1:raise ValueError('Stage epoch limits must be positive.')


def prepare(source,settings,stage,root):
    source=source[:source.index('if GUIInfo.enabled():')]
    reference=(root/'normal_strain_reference.py').read_text(encoding='utf-8-sig')
    assert source.count('\ndef net_force(')==1
    source=source.replace('\ndef net_force(',reference+'\ndef net_force(')
    def once(a,b):
        nonlocal source
        assert source.count(a)==1,(a,source.count(a))
        source=source.replace(a,b)
    count=active_count(settings,stage)
    mapping='\n'.join(f"joint_diff_{i}=mesh.edges().create('diff/{name}',-1)\nview(joint_diff_{i})[:]={i}" for i,name in enumerate(ATTRS[:count]))
    once("diff_friction_k: AttributeSlotI32 = mesh.edges().create('diff/friction_stiffness', -1)\nview(diff_friction_k)[:] = 0",mapping)
    once('parm = MyDiffSimParameter(scene=scene, size=1)',f'parm = MyDiffSimParameter(scene=scene, size={count})')
    once('    u = np.sqrt(BENDING_FRICTION)   \n    parm.U().fill_(u)',
        '    parm.U().copy_(torch.sqrt(torch.tensor(_initial_active, dtype=torch.float64)))')
    once('    parm_bending_friction_k_tensor = optimizer.param_groups[0]["params"][0]**2',
        '    joint_p = parm.forward().reshape(-1)\n    if joint_p.numel()==3:\n        joint_p=torch.cat((joint_p,torch.tensor([Hardening_K,YEILD_THETA],dtype=torch.double)))\n    parm_bending_friction_k_tensor=joint_p[1]\n    thres_thetas_tensor=joint_p[2]*torch.ones_like(thres_thetas_tensor)')
    once('        force = net_force(X_pred, BENDING_STIFFNESS, parm_bending_friction_k_tensor, Hardening_K, rest_yield_theta_ori_tensor,',
        '        force = net_force(X_pred, joint_p[0], parm_bending_friction_k_tensor, joint_p[3], joint_p[4]*torch.ones_like(rest_yield_theta_ori_tensor),')
    once('    for f in range(TRAIN_START_FRAME, TRAIN_END_FRAME):','    for f in _loss_frame_indices:')
    once('    loss.backward()', '    _capture_bend_epoch(joint_p,forces_pred,loss)\n    loss.backward()')
    hook='''optimizer = optim.Adam(diff_module.parameters(),lr=1.0)
joint_rates=torch.tensor(_active_rates,dtype=torch.double)
joint_before=None
def joint_pre_step(optimizer,args,kwargs):
    global joint_before
    joint_before=parm.U().detach().clone()
def joint_post_step(optimizer,args,kwargs):
    with torch.no_grad():
        parm.U().copy_(joint_before+joint_rates*(parm.U()-joint_before))
optimizer.register_step_pre_hook(joint_pre_step)
optimizer.register_step_post_hook(joint_post_step)'''
    once('optimizer = optim.Adam(diff_module.parameters(), lr=PARAM_LR)',hook)
    axis=direction(stage);data=np.load(settings[axis+'_data'],allow_pickle=False)
    p=settings['parameters'];m=settings['material'];end=70 if stage==1 else 150
    values=dict(Warp_Stiffness=settings['warp'],Weft_Stiffness=settings['weft'],MASS_DENSITY=settings['density'],THICKNESS=settings['thickness'],
        K33_0=m['k33_0'],K33_1=m['k33_1'],Fric_K33=m['friction'],Hardening_K33=m['hardening'],Thres_uv=100.,Yield_UV=100.,
        BENDING_STIFFNESS=p[0],BENDING_FRICTION=p[1],THRES_THETA=p[2],Hardening_K=p[3],YEILD_THETA=p[4],
        tested_fabric_type=settings['fabric'],MESH_PATH=str(root/'mesh.obj'),Up_Direction=axis,EndFrame=end,
        TRAIN_START_FRAME=29,TRAIN_END_FRAME=end,COMP_DIS=float(data[0].max())/1000,PARAM_LR=1.,NUM_EPOCH=settings['epochs'][stage-1])
    tree=ast.parse(source)
    found=set()
    body=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in values:
            name=node.targets[0].id;node.value=ast.parse(repr(values[name]),mode='eval').body;found.add(name)
        if isinstance(node,ast.With) and any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id=='gt_comp_force' for n in ast.walk(node)):
            node=ast.parse('gt_comp_force=torch.tensor(_real_gt,dtype=torch.double)').body[0]
        body.append(node)
    assert found==set(values),set(values)-found
    tree.body=body
    class Quiet(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func) in ('print','Timer.report'):
                return ast.copy_location(ast.Pass(),node)
            return self.generic_visit(node)
    tree=Quiet().visit(tree)
    source=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    source=source.replace('scene = Scene(config)',
        f"config['contact']['enable']={settings['contact']!r}\nconfig['newton']['velocity_tol']={settings['newton']!r}\nconfig['linear_system']['tol_rate']={settings['linear']!r}\nscene = Scene(config)")
    compile(source,'<bending scene>','exec')
    return source,data


class Recorder:
    def __init__(self,root):self.root=root;self.directory=None;self.best={};self.count={}
    def save(self,row,settings,source):
        if self.directory is None:
            self.directory=self.root/'logs'/('gui_bending_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.directory.mkdir(parents=True,exist_ok=False)
        stage=row['stage'];folder=self.directory/f'stage{stage}_{direction(stage)}';folder.mkdir(exist_ok=True)
        if not (folder/'config.json').exists():
            (folder/'scene_source.py').write_text(source,encoding='utf-8')
            (folder/'config.json').write_text(json.dumps(dict(settings=settings,loss_frames=frames(stage),
                parameter_order=NAMES,active_parameters=NAMES[:active_count(settings,stage)],plasticity_enabled=settings['plastic'],parameterization='P=U^2; original post-step scaling and DiffSimLR retained',
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                data_sha256=hashlib.sha256(Path(settings[direction(stage)+'_data']).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
        values={k:row[k] for k in ('pred','gt','displacement_mm','parameters','loss','epoch','stage','seconds')}
        np.savez_compressed(folder/f'epoch_{row["epoch"]:05d}.npz',**values)
        with (folder/'epochs.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({k:v for k,v in row.items() if k not in ('pred','gt','displacement_mm')},allow_nan=False)+'\n');stream.flush()
        if row['loss']<self.best.get(stage,float('inf')):
            np.savez_compressed(folder/'best_result.npz',**values);self.best[stage]=row['loss']
        self.count[stage]=self.count.get(stage,0)+1
        (folder/'summary.json').write_text(json.dumps(dict(saved_epochs=self.count[stage],best_saved_loss=self.best[stage],last_saved_epoch=row['epoch']),indent=2),encoding='utf-8')


class Controller:
    def __init__(self,root,settings):
        if type(settings.get('plastic')) is not bool:raise ValueError('Explicit plasticity setting required')
        if not settings['plastic']:settings['parameters'][3:]=[50.,100.]
        self.root=root;self.settings=settings;self.stage=1;self.running=False;self.finished=False;self.pending=None
        self.save_records=True;self.save_error='';self.history=[];self.best_by_stage={};self.message='Ready';self.error=''
        from bending_best_io import BestStore
        self.best_store=BestStore(root,settings)
        self.records=Recorder(root);self.records.directory=self.best_store.directory
        self.log_axis=False;self.follow=True;self.curve_index=1;self.env=None
    def attach(self,env,scene,data):
        self.env=env;self.scene_source=scene;self.data=data;self.epoch=0;self.best_loss=float('inf')
        self.best_parameters=list(self.settings['parameters']);self.reference=float('inf');self.stale=0;self.stage_times=[];self.capture=None
        def capture(P,pred,loss):
            self.capture=dict(parameters=P.detach().cpu().tolist(),pred=pred.detach().cpu().numpy().copy())
            self.best_store.consider(self.stage,self.epoch+1,self.capture['parameters'],float(loss.detach()),
                self.capture['pred'],env['gt_comp_force'].detach().cpu().numpy(),self.data[0,frames(self.stage)])
        env['_capture_bend_epoch']=capture
        with torch.no_grad():
            env['lr']._old_U.copy_(env['parm'].U())
            env['lr']._old_P.copy_(env['parm'].forward())
    def start(self):
        if self.finished:self.message='Finished. Apply settings to start again.';return
        self.env['run']=False;self.running=True;self.defer=True;self.message='Training'
    def tick(self):
        if not self.running:return
        if self.defer:self.defer=False;return
        e=self.env
        try:
            self.capture=None;t=time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):loss=float(e['optimizer'].step(e['loss_fn']).detach())
            seconds=time.perf_counter()-t;assert self.capture is not None
            row=dict(self.capture);gt=e['gt_comp_force'].detach().cpu().numpy().copy()
            assert np.isfinite(loss) and np.isfinite(row['pred']).all() and math.isclose(float(np.mean((row['pred']-gt)**2)),loss,rel_tol=1e-6,abs_tol=1e-12)
            after=e['parm'].forward().detach().cpu().tolist()
            if active_count(self.settings,self.stage)==3:after+=self.settings['parameters'][3:]
            plastic_change=None
            if not self.settings['plastic']:
                history=np.asarray(e['diff_module'].get_plastic_thetas())
                rest=e['rest_plastic_thetas']
                plastic_change=float(np.max(np.abs(history-rest)))
                if plastic_change>1e-12:raise RuntimeError('Plasticity is disabled but a nonzero plastic increment was recorded')
            self.epoch+=1
            row.update(plasticity_enabled=self.settings['plastic'],plastic_max_change=plastic_change,stage=self.stage,direction=direction(self.stage),epoch=self.epoch,loss=loss,gt=gt,seconds=seconds,
                displacement_mm=self.data[0,frames(self.stage)].copy(),frames=frames(self.stage),after_step=after,
                gradient_U=e['parm'].U().grad.detach().cpu().tolist())
            self.history.append(row);self.stage_times.append(seconds)
            if loss<self.best_loss:self.best_loss=loss;self.best_parameters=list(row['parameters'])
            if loss<self.reference*.999:self.reference=loss;self.stale=0
            else:self.stale+=1
            if self.save_records:
                try:self.records.save(row,self.settings,self.scene_source);self.save_error=''
                except Exception as exc:self.save_error='Saving failed: '+str(exc);self.save_records=False;traceback.print_exc()
            plateau=self.settings['early_stop'] and self.epoch>=30 and self.stale>=20
            if self.epoch>=self.settings['epochs'][self.stage-1] or plateau:
                self.running=False;self.best_by_stage[self.stage]=dict(loss=self.best_loss,parameters=list(self.best_parameters))
                self.message='Plateau early stop' if plateau else 'Epoch limit reached'
                if self.stage<3:
                    self.settings['parameters']=list(self.best_parameters);self.stage+=1;self.pending='next_stage'
                else:
                    self.finished=True
                    with torch.no_grad():
                        e['parm'].U().copy_(torch.tensor(self.best_parameters[:active_count(self.settings,self.stage)],dtype=torch.double).sqrt());e['parm'].sync()
                        e['lr']._old_U.copy_(e['parm'].U());e['lr']._old_P.copy_(e['parm'].forward())
                    self.best_store.mark('completed')
                    self.message+='; weft best parameters restored (not replayed)'
        except Exception as exc:self.running=False;self.error=str(exc);self.message='Training failed';traceback.print_exc()
    def draw(self):
        e=self.env;imgui.Text(f'Isotropic bending: stage {self.stage}/3, {direction(self.stage)}')
        imgui.Text('Bending plasticity: '+('ON' if self.settings['plastic'] else 'OFF (yield 100, hardening 50 fixed)'))
        imgui.Text(f'Active parameters: {active_count(self.settings,self.stage)}')
        _,self.save_records=imgui.Checkbox('Save training records',self.save_records);set_logging_theme(self.save_records)
        if not self.save_records:imgui.Text('DETAILED LOGS OFF - best parameters are still saved')
        imgui.Text('Best parameters (always saved): '+str(self.best_store.directory))
        if self.save_error:imgui.Text(self.save_error)
        imgui.Text(f'Epoch {self.epoch}/{self.settings["epochs"][self.stage-1]}; best stage MSE: {self.best_loss:.9g}')
        imgui.Text(self.message)
        if self.error:imgui.Text(self.error)
        if self.stage_times:
            avg=float(np.mean(self.stage_times));imgui.Text(f'Last {self.stage_times[-1]:.2f} s/epoch; mean {avg:.2f} s/epoch')
            imgui.Text(f'Current-stage remaining compute: {max(0,self.settings["epochs"][self.stage-1]-self.epoch)*avg/60:.1f} min (may early stop)')
        p=e['parm'].forward().detach().cpu().tolist()
        if len(p)==3:p+=self.settings['parameters'][3:]
        for name,value in zip(NAMES,p):imgui.Text(f'{name}: {value:.9g}')
        for stage,result in self.best_by_stage.items():imgui.Text(f'Stage {stage} {direction(stage)} best: {result["loss"]:.9g}; P={result["parameters"]}')
        _,self.log_axis=imgui.Checkbox('Log10 loss axes',self.log_axis)
        for stage in (1,2,3):
            values=np.array([r['loss'] for r in self.history if r['stage']==stage])
            if len(values):
                if self.log_axis:values=np.log10(np.maximum(values,1e-300))
                imgui.PlotLines(f'##bending_stage{stage}',values.astype(np.float32),overlay_text=f'Stage {stage} {direction(stage)} MSE',graph_size=(300.,120.))
        if self.history:
            _,self.follow=imgui.Checkbox('Show latest epoch force',self.follow)
            if self.follow:self.curve_index=len(self.history)
            else:
                _,self.curve_index=imgui.InputInt('Force curve record',self.curve_index);self.curve_index=max(1,min(len(self.history),self.curve_index))
            r=self.history[self.curve_index-1];imgui.Text(f'Force: stage {r["stage"]} {r["direction"]}, epoch {r["epoch"]}, MSE={r["loss"]:.9g}')
            draw_force_overlay(r['displacement_mm'],r['gt'],r['pred'],break_after=41 if r['stage']>1 else None)
        if self.running and imgui.Button('Pause optimization'):self.running=False;self.message='Paused'
        imgui.Text('Window refreshes between full epochs. Applying settings resets all three stages.')


class Panel:
    def __init__(self,c,presets):self.c=c;self.v=json.loads(json.dumps(c.settings));self.presets=presets;self.error=''
    def draw(self):
        v=self.v;c=self.c;choices=list(self.presets);idx=choices.index(v['fabric'])
        changed,idx=imgui.Combo('Material',idx,choices)
        if changed:
            name=choices[idx];m=self.presets[name];v.update(fabric=name,material=m,warp=m['warp'],weft=m['weft'],density=m['density'],thickness=m['thickness'])
            for axis in ('warp','weft'):v[axis+'_data']=str(c.root/'data'/f'{name}_{axis}_compression.npy')
        for axis in ('warp','weft'):_,v[axis+'_data']=imgui.InputText(axis+' GT (.npy)',v[axis+'_data'],max_str_len=4096)
        for name,label in [('density','Density (kg/m^3)'),('thickness','Thickness (m)'),('warp','Fixed warp stretch (Pa)'),('weft','Fixed weft stretch (Pa)'),('newton','Newton velocity tolerance'),('linear','Linear relative tolerance')]:
            _,v[name]=imgui.InputFloat(label,float(v[name]),format='%.8g')
        changed,v['plastic']=imgui.Checkbox('Activate bending plasticity (isotropic)',v['plastic'])
        if changed:
            if v['plastic']:v['parameters'][3:]=v.get('plastic_initial_values',[50.,1.])
            else:
                v['plastic_initial_values']=v['parameters'][3:].copy()
                v['parameters'][3:]=[50.,100.]
        for i,name in enumerate(NAMES):
            if i>=3 and not v['plastic']:
                imgui.Text(f'{name}: fixed at {v["parameters"][i]:g} (plasticity OFF)')
                continue
            _,v['parameters'][i]=imgui.InputFloat('Initial '+name,float(v['parameters'][i]),format='%.8g')
            _,v['rates'][i]=imgui.InputFloat('Learning rate U: '+name,float(v['rates'][i]),format='%.8g')
        if v['plastic']:v['plastic_initial_values']=v['parameters'][3:].copy()
        imgui.Text('This choice is saved with both directions and controls the subsequent joint model.')
        for i in range(3):_,v['epochs'][i]=imgui.InputInt(f'Stage {i+1} maximum epochs',v['epochs'][i])
        _,v['early_stop']=imgui.Checkbox('Plateau early stop (minimum 30, patience 20)',v['early_stop'])
        _,v['contact']=imgui.Checkbox('Forward collision',v['contact'])
        imgui.Text('P=U squared. Stage 1: warp 30-70. Stages 2/3: 30-70 and 120-150, full rollout to 150.')
        imgui.Text('Shear sliding/plasticity is disabled (threshold and yield fixed 100); soft constraint is 200.')
        if imgui.Button('Apply settings and reset all stages'):
            try:validate(v);c.pending=('reset',json.loads(json.dumps(v)));c.running=False
            except Exception as exc:self.error=str(exc)
        if self.error:imgui.Text(self.error)


def gui_code(source):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.If) and ast.unparse(n.test)=='GUIInfo.enabled()')
    body=[n for n in node.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func) in ('ps.init','ps.show'))]
    for n in body:
        if isinstance(n,ast.FunctionDef) and n.name=='update':
            index=next(i for i,v in enumerate(n.body) if not isinstance(v,ast.Global))
            n.body[index:index]=ast.parse('panel.draw()\ncontroller.draw()\nif controller.pending:\n    ps.unshow()\n    return').body
            n.body.extend(ast.parse('controller.tick()\nif controller.pending:\n    ps.unshow()').body)
            for v in n.body:
                if isinstance(v,ast.If) and ast.unparse(v.test)=="imgui.Button('Run & Stop')":v.body[:0]=ast.parse('controller.running=False').body
    class Change(ast.NodeTransformer):
        def visit_Expr(self,n):
            if isinstance(n.value,ast.Call) and ast.unparse(n.value.func) in ('print','Timer.report'):return ast.Pass()
            return self.generic_visit(n)
        def visit_Assign(self,n):
            if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='net_force':
                before=ast.parse('''_preview_p=parm.forward().detach()
if _preview_p.numel()==3:
    _preview_p=torch.cat((_preview_p,torch.tensor([Hardening_K,YEILD_THETA],dtype=torch.double)))
''').body
                for arg,i in [(1,0),(2,1),(3,3),(4,4),(8,2)]:n.value.args[arg]=ast.parse(f'_preview_p[{i}]',mode='eval').body
                return before+[n]
            return self.generic_visit(n)
    tree=Change().visit(ast.Module(body=body,type_ignores=[]))
    return compile(ast.fix_missing_locations(tree),'<bending gui>','exec')


def run(model,args):
    torch.set_num_threads(4);model=Path(model).resolve();root=model.parent;source=model.read_text(encoding='utf-8-sig')
    presets=json.loads((root/'materials.json').read_text());m=presets[args.fabric]
    settings=dict(fabric=args.fabric,material=m,warp=m['warp'],weft=m['weft'],density=m['density'],thickness=m['thickness'],
        plastic=not args.no_plastic,plastic_initial_values=[50.,1.],parameters=[600.,400.,.1,50.,1.],rates=[.3,.3,.002,.1,.001],epochs=[args.epochs]*3,early_stop=True,
        newton=.005,linear=.0001,contact=False,
        **{a+'_data':str(root/'data'/f'{args.fabric}_{a}_compression.npy') for a in ('warp','weft')})
    if not settings['plastic']:settings['parameters'][3:]=[50.,100.]
    validate(settings);controller=Controller(root,settings);headless=args.headless_train
    if not headless:ps.init()
    graphics=gui_code(source)
    while True:
        scene,data=prepare(source,controller.settings,controller.stage,root)
        selection=frames(controller.stage);size=active_count(controller.settings,controller.stage)
        env=dict(__name__='_managed_bending',__file__=str(model),_real_gt=data[1,selection].tolist(),
            _initial_active=controller.settings['parameters'][:size],_active_rates=controller.settings['rates'][:size],
            _loss_frame_indices=[f-1 for f in selection])
        try:
            with contextlib.redirect_stdout(io.StringIO()):exec(compile(scene,str(model),'exec'),env)
            controller.attach(env,scene,data);panel=Panel(controller,presets)
            env.update(controller=controller,panel=panel);env['optimize_once']=controller.start
            if controller.stage>1:controller.start()
            if headless:
                controller.start()
                while controller.running:controller.tick()
                if controller.error:raise RuntimeError(controller.error)
            else:exec(graphics,env);ps.show()
        finally:
            if not headless:set_logging_theme(True);ps.set_user_callback(lambda:None);ps.remove_all_structures()
            if controller.pending is None and not controller.finished:
                controller.best_store.mark('interrupted')
            controller.env=None;env.clear();gc.collect()
        pending=controller.pending;controller.pending=None
        if pending=='next_stage':continue
        if isinstance(pending,tuple):
            enabled=controller.save_records;controller=Controller(root,pending[1]);controller.save_records=enabled;continue
        break
