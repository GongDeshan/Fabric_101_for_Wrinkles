"""Joint GUI with strict isotropic best loading and serial, contained cold workers."""
import argparse,datetime,hashlib,io,json,math,os,shutil,subprocess,sys,time,traceback,uuid
from pathlib import Path
import numpy as np
import torch
import polyscope as ps
from polyscope import imgui
from bending_best_io import load_isotropic_run,atomic_bytes,atomic_json,digest
from stretch_logging_theme import set_logging_theme
from stretch_force_plot import draw_force_overlay
from windows_training_lifetime_copy import Lifetime

ROOT=Path(__file__).resolve().parent
NAMES=('kb_warp','kb_weft','friction_warp','friction_weft','hardening_warp','hardening_weft','yield_warp','yield_weft','thres_warp','thres_weft')
SIX=(0,1,2,3,8,9)
RATES=[.3,.3,.3,.3,.1,.1,.001,.001,.002,.002]
FRAMES=list(range(30,71))+list(range(120,151))

def initial_parameters(loaded,plastic):
    result={}
    for axis in ('warp','weft'):
        p=loaded['bests'][axis]
        for name,value in zip(('kb','friction','thres','hardening','yield'),p):result[name+'_'+axis]=float(value)
    if not plastic:
        for axis in ('warp','weft'):
            if result['hardening_'+axis]!=50. or result['yield_'+axis]!=100.:raise ValueError('No-plastic isotropic log has inconsistent plastic parameters')
    return result


class Controller:
    def __init__(self):
        self.path='';self.loaded=None;self.error='';self.message='Select an isotropic run and load its best parameters.'
        self.settings=dict(plastic=None,newton=.005,linear=.0001,contact=True,epochs=100,early_stop=True,rates=list(RATES))
        self.run_dir=None;self.running=False;self.finished=False;self.pause_requested=False;self.child=None;self.log=None
        self.results=[];self.rows=[];self.completed=0;self.best_loss=float('inf');self.best_p=None;self.stale=0;self.reference=float('inf')
        self.save_records=True;self.save_error='';self.follow=True;self.curve_record=1;self.curve_axis=0;self.log_axis=False
        self.life=Lifetime(ROOT/'joint_gui.lock')
        self.mesh_handle=None
    def load(self):
        if self.child is not None:raise RuntimeError('Pause and finish the active epoch before loading another run.')
        if self.running:raise RuntimeError('Pause training before loading another run.')
        self.loaded=None
        loaded=load_isotropic_run(self.path,ROOT/'mesh.obj')
        s=loaded['manifest']['settings']
        self.loaded=loaded
        try:
            self.settings.update({k:s[k] for k in ('density','thickness','warp','weft','material','fabric')})
            self.settings['plastic']=loaded['plasticity_enabled']
            self.configure()
        except BaseException:
            self.loaded=None
            raise
        self.draft=json.loads(json.dumps(self.settings))
        self.message='Loaded and verified both direction bests. History-position gradient is OFF.'
    def configure(self):
        if self.loaded is None:raise RuntimeError('Load a valid isotropic run first; no default parameters are available.')
        if self.child is not None:raise RuntimeError('Pause and finish the active epoch before applying settings.')
        s=self.settings
        if s['plastic'] is not self.loaded['plasticity_enabled']:raise ValueError('Joint plasticity must follow the isotropic log; change it in isotropic training')
        numbers=[s[k] for k in ('density','thickness','newton','linear')]+s['rates']
        if not np.isfinite(numbers).all() or min(numbers)<=0 or s['epochs']<1:raise ValueError('Settings must be positive and finite.')
        if self.run_dir is not None:
            self.manifest['lifecycle']='reset';self.save_manifest()
        self.indices=list(range(10)) if s['plastic'] else list(SIX)
        self.initial=initial_parameters(self.loaded,s['plastic'])
        self.u=torch.nn.Parameter(torch.tensor([self.initial[NAMES[i]] for i in self.indices],dtype=torch.double).sqrt())
        self.optimizer=torch.optim.Adam([self.u],lr=1.)
        self.run_dir=None;self.running=False;self.finished=False;self.pause_requested=False;self.results=[];self.rows=[];self.completed=0
        self.best_loss=float('inf');self.best_p=None;self.reference=float('inf');self.stale=0;self.error='';self.save_error=''
    def physical(self):
        p=dict(self.initial);p.update(zip([NAMES[i] for i in self.indices],self.u.detach().square().tolist()))
        if not np.isfinite(list(p.values())).all():raise RuntimeError('Nonfinite physical parameters')
        return p
    def signed(self,p):
        result=[math.sqrt(p[n]) for n in NAMES]
        for index,value in zip(self.indices,self.u.detach().tolist()):result[index]=value
        return result
    def ensure_directory(self):
        if self.run_dir is not None:return
        self.run_dir=ROOT/'logs'/('gui_joint_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.run_dir.mkdir(parents=True,exist_ok=False)
        atomic_json(self.run_dir/'input_isotropic_manifest.json',self.loaded['manifest'])
        self.manifest=dict(schema='orthotropic-joint-best-v1',lifecycle='running',history_gradient=False,
            isotropic_run=self.loaded['directory'],isotropic_run_id=self.loaded['manifest']['run_id'],
            isotropic_manifest_sha256=digest(self.run_dir/'input_isotropic_manifest.json'),
            parameter_order=NAMES,active_parameters=[NAMES[i] for i in self.indices],initial=self.initial,
            settings=json.loads(json.dumps(self.settings)),loss_frames=FRAMES,forward_end_frame=150,
            plastic_override=None,plasticity_source='isotropic run manifest',plasticity_enabled=self.settings['plastic'],
            mesh_sha256=digest(ROOT/'mesh.obj'),source_hashes={p.name:digest(p) for p in ROOT.glob('*.py')},best=None)
        self.save_manifest()
    def save_manifest(self):
        if self.run_dir is not None:atomic_json(self.run_dir/'run_manifest.json',self.manifest)
    def start(self):
        if self.loaded is None:raise RuntimeError('No isotropic log has been loaded. Training is disabled.')
        if self.finished:raise RuntimeError('Training finished. Apply settings to start another run.')
        self.ensure_directory();self.running=True;self.pause_requested=False;self.message='Training'
    def spawn(self):
        self.ensure_directory()
        if not self.results:
            self.evaluated=self.physical();self.evaluated_signed=self.signed(self.evaluated);self.epoch_started=time.perf_counter()
        axis='warp' if not self.results else 'weft'
        for name in ('warp','weft'):
            info=self.loaded['manifest']['data'][name]
            if digest(info['path'])!=info['sha256']:raise RuntimeError('Isotropic GT changed after loading')
        if digest(ROOT/'mesh.obj')!=self.loaded['manifest']['mesh']['sha256']:raise RuntimeError('Joint mesh changed after loading')
        out=self.run_dir/f'epoch_{self.completed+1:05d}_{axis}';(out/'snapshot').mkdir(parents=True,exist_ok=False)
        for p in ROOT.glob('*.py'):shutil.copy2(p,out/'snapshot'/p.name)
        request=dict(direction=axis,parameters=self.evaluated,signed_U=self.evaluated_signed,gradient=True,
            active=[NAMES[i] for i in self.indices],settings=self.settings,fabric=self.settings['fabric'],
            data_path=self.loaded['manifest']['data'][axis]['path'])
        atomic_json(out/'request.json',request)
        self.log=(out/'worker.log').open('w',encoding='utf-8')
        env=os.environ.copy()
        for k in ('UIPC_NEWTON_AUDIT','UIPC_FRAME_TRACE'):env.pop(k,None)
        self.child=subprocess.Popen([str(Path(sys.executable).with_name('python.exe')),'-u',str(ROOT/'joint_worker.py'),
            str(out/'request.json'),str(out/'result.json')],cwd=ROOT,stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,
            text=True,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            self.life.assign(self.child);self.child.stdin.write('START\n');self.child.stdin.flush()
        except BaseException:
            if self.child.poll() is None:self.child.kill()
            self.child.wait(timeout=10);self.child.stdin.close();self.log.close();self.child=None;self.log=None;raise
        self.current_folder=out;self.worker_started=time.perf_counter();self.message=f'Epoch {self.completed+1}: evaluating {axis}'
    def commit_best(self,loss):
        if loss>=self.best_loss:return
        revision='best_'+uuid.uuid4().hex
        values=dict(parameters=np.array([self.evaluated[n] for n in NAMES]),parameter_order=NAMES,epoch=self.completed+1,
            loss=loss,warp_loss=self.results[0]['loss'],weft_loss=self.results[1]['loss'],frames=FRAMES,
            warp_prediction=self.results[0]['prediction'],weft_prediction=self.results[1]['prediction'],
            warp_gt=self.results[0]['gt'],weft_gt=self.results[1]['gt'])
        buffer=io.BytesIO();np.savez_compressed(buffer,**values)
        curve=self.run_dir/(revision+'.npz');atomic_bytes(curve,buffer.getvalue())
        payload=dict(epoch=self.completed+1,loss=loss,parameters=self.evaluated,curve_file=curve.name,curve_sha256=digest(curve))
        file=self.run_dir/(revision+'.json');atomic_json(file,payload)
        self.best_loss=loss;self.best_p=dict(self.evaluated)
        self.manifest['best']=dict(file=file.name,sha256=digest(file),loss=loss)
        self.save_manifest()
    def finish_epoch(self):
        loss=float(np.mean([r['loss'] for r in self.results]));gradient=np.mean([r['gradient_U'] for r in self.results],axis=0)
        if not math.isfinite(loss) or not np.isfinite(gradient).all():raise RuntimeError('Nonfinite joint loss/gradient')
        self.commit_best(loss) 
        self.optimizer.zero_grad(set_to_none=True);self.u.grad=torch.tensor(gradient[self.indices],dtype=torch.double)
        before=self.u.detach().clone();self.optimizer.step()
        with torch.no_grad():self.u.copy_(before+torch.tensor([self.settings['rates'][i] for i in self.indices])*(self.u-before))
        self.completed+=1
        row=dict(epoch=self.completed,loss=loss,warp_loss=self.results[0]['loss'],weft_loss=self.results[1]['loss'],
            parameters=dict(self.evaluated),after_step=self.physical(),gradient_U=gradient.tolist(),
            seconds=time.perf_counter()-self.epoch_started,results=self.results)
        self.rows.append(row)
        if loss<self.reference*.999:self.reference=loss;self.stale=0
        else:self.stale+=1
        if self.save_records:
            try:
                scalar={k:v for k,v in row.items() if k!='results'}
                with (self.run_dir/'epochs.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(scalar,allow_nan=False)+'\n');stream.flush()
                np.savez_compressed(self.run_dir/f'epoch_{self.completed:05d}.npz',
                    warp_prediction=self.results[0]['prediction'],weft_prediction=self.results[1]['prediction'],
                    warp_gt=self.results[0]['gt'],weft_gt=self.results[1]['gt'],frames=FRAMES,
                    parameters=[self.evaluated[n] for n in NAMES],loss=loss)
                self.save_error=''
            except Exception as exc:self.save_error='Detailed log save failed: '+str(exc);self.save_records=False;traceback.print_exc()
        self.manifest.update(last_completed_epoch=self.completed,last_loss=loss);self.save_manifest()
        self.results=[]
        plateau=self.settings['early_stop'] and self.completed>=30 and self.stale>=20
        if self.completed>=self.settings['epochs'] or plateau:
            self.running=False;self.finished=True;self.manifest['lifecycle']='completed';self.save_manifest()
            self.message='Plateau early stop' if plateau else 'Epoch limit reached'
        elif self.pause_requested:self.running=False;self.message='Paused after completed epoch'
    def tick(self):
        try:
            if self.child is not None:
                if self.child.poll() is None:
                    if time.perf_counter()-self.worker_started>3600:raise TimeoutError('Directional worker exceeded one hour')
                    return
                code=self.child.returncode;self.child.wait(timeout=10);self.child.stdin.close();self.log.close();self.log=None;self.child=None
                if code:raise RuntimeError(f'Worker failed ({code}); see {self.current_folder/"worker.log"}')
                result=json.loads((self.current_folder/'result.json').read_text())
                if result['parameters']!=self.evaluated or result['signed_U']!=self.evaluated_signed:raise RuntimeError('Worker parameter/sign mismatch')
                if result['history_gradient'] is not False:raise RuntimeError('History-gradient switch is not OFF')
                marker='[ortho-history-gradient-v1] history-position adjoint=OFF (approximate)'
                if marker not in (self.current_folder/'worker.log').read_text(encoding='utf-8',errors='replace'):raise RuntimeError('Native history-OFF marker missing')
                gt=np.load(self.loaded['manifest']['data']['warp' if not self.results else 'weft']['path'],allow_pickle=False)[1,FRAMES]
                if not np.array_equal(gt,result['gt']):raise RuntimeError('Worker GT/frame mismatch')
                if not math.isclose(float(np.mean((np.array(result['prediction'])-gt)**2)),result['loss'],rel_tol=1e-6,abs_tol=1e-12):raise RuntimeError('Worker curve/loss mismatch')
                self.results.append(result)
                if self.mesh_handle is not None:
                    with np.load(self.current_folder/'last_mesh.npz') as mesh:self.mesh_handle.update_vertex_positions(mesh['vertices'])
                if len(self.results)==2:self.finish_epoch()
            if self.running and self.child is None:self.spawn()
        except Exception as exc:
            self.error=str(exc);self.message='Training stopped';self.running=False
            if self.run_dir is not None:self.manifest.update(lifecycle='failed',error=self.error);self.save_manifest()
            self.stop_child();traceback.print_exc()
    def stop_child(self):
        if self.child is not None:
            if self.child.poll() is None:self.child.kill()
            self.child.wait(timeout=10)
            if self.child.stdin:self.child.stdin.close()
            self.child=None
        if self.log:self.log.close();self.log=None
    def close(self):
        try:
            self.stop_child()
            if self.run_dir is not None and not self.finished and self.manifest.get('lifecycle')!='failed':
                self.manifest['lifecycle']='interrupted';self.save_manifest()
        finally:self.life.close()

    def draw(self):
        imgui.Text('Orthotropic joint bending - history-position gradient OFF')
        _,self.path=imgui.InputText('Isotropic run folder',self.path,max_str_len=4096)
        if imgui.Button('Load isotropic warp and weft bests'):
            try:self.load();self.error=''
            except Exception as exc:self.error=str(exc)
        if self.loaded:
            imgui.Text('Verified material: '+self.settings['fabric'])
            for axis,p in self.loaded['bests'].items():imgui.Text(axis+' imported: '+str(p))
            imgui.Text('Settings below apply when you press Apply; changing them during training is blocked.')
            if not hasattr(self,'draft'):self.draft=json.loads(json.dumps(self.settings))
            d=self.draft
            imgui.Text('Bending plasticity from isotropic log: '+('ON' if self.loaded['plasticity_enabled'] else 'OFF'))
            if not d['plastic']:imgui.Text('Plasticity OFF: yield=100, hardening=50 fixed; six trainable parameters.')
            for name,label in [('density','Density (kg/m^3)'),('thickness','Thickness (m)'),('newton','Newton velocity tolerance'),('linear','Linear relative tolerance')]:
                _,d[name]=imgui.InputFloat(label,float(d[name]),format='%.8g')
            for i,name in enumerate(NAMES):_,d['rates'][i]=imgui.InputFloat('LR U '+name,float(d['rates'][i]),format='%.8g')
            _,d['epochs']=imgui.InputInt('Maximum joint epochs',d['epochs'])
            _,d['early_stop']=imgui.Checkbox('Plateau early stop (min 30 / patience 20)',d['early_stop'])
            _,d['contact']=imgui.Checkbox('Forward collision',d['contact'])
            if imgui.Button('Apply settings and reset joint training'):
                try:
                    if self.child is not None or self.running:raise RuntimeError('Pause at an epoch boundary before applying settings.')
                    previous=self.settings
                    self.settings=json.loads(json.dumps(d))
                    try:self.configure()
                    except BaseException:self.settings=previous;raise
                    self.message='Joint training reset'
                except Exception as exc:self.error=str(exc)
            if imgui.Button('optim'):
                try:self.start();self.error=''
                except Exception as exc:self.error=str(exc)
        _,self.save_records=imgui.Checkbox('Save detailed training records',self.save_records);set_logging_theme(self.save_records)
        if not self.save_records:imgui.Text('DETAILED LOGS OFF - best parameters are always saved')
        if self.run_dir:imgui.Text('Records / durable best: '+str(self.run_dir))
        if self.save_error:imgui.Text(self.save_error)
        if self.error:imgui.Text('ERROR: '+self.error)
        imgui.Text(self.message)
        imgui.Text(f'Epoch {self.completed}/{self.settings["epochs"]}; best mean MSE {self.best_loss:.10g}')
        if self.running and imgui.Button('Pause after current epoch'):self.pause_requested=True;self.message='Pause requested; current paired epoch will finish'
        if self.loaded:
            for name,value in self.physical().items():imgui.Text(f'{name}: {value:.9g}')
        if self.rows:
            last=self.rows[-1];avg=float(np.mean([r['seconds'] for r in self.rows]))
            imgui.Text(f'Last {last["seconds"]:.2f} s/epoch; mean {avg:.2f} s/epoch; remaining compute {max(0,self.settings["epochs"]-self.completed)*avg/60:.1f} min')
            imgui.Text(f'Warp MSE {last["warp_loss"]:.10g}; weft MSE {last["weft_loss"]:.10g}; mean {last["loss"]:.10g}')
            _,self.log_axis=imgui.Checkbox('Log10 loss axes',self.log_axis)
            for key,label in [('loss','Mean MSE'),('warp_loss','Warp MSE'),('weft_loss','Weft MSE')]:
                values=np.array([r[key] for r in self.rows]);values=np.log10(np.maximum(values,1e-300)) if self.log_axis else values
                imgui.PlotLines('##'+key,values.astype(np.float32),overlay_text=label,graph_size=(300.,110.))
            _,self.follow=imgui.Checkbox('Show latest epoch curves',self.follow)
            if self.follow:self.curve_record=len(self.rows)
            else:
                _,self.curve_record=imgui.InputInt('Curve epoch',self.curve_record);self.curve_record=max(1,min(len(self.rows),self.curve_record))
            _,self.curve_axis=imgui.Combo('Curve direction',self.curve_axis,['warp','weft'])
            row=self.rows[self.curve_record-1];axis=['warp','weft'][self.curve_axis];result=row['results'][self.curve_axis]
            imgui.Text(f'Curve epoch {row["epoch"]} {axis}; MSE {result["loss"]:.9g}')
            data=np.load(self.loaded['manifest']['data'][axis]['path'],allow_pickle=False)
            draw_force_overlay(data[0,FRAMES],result['gt'],result['prediction'],break_after=41)


def main():
    parser=argparse.ArgumentParser(description='History-off joint bending initialized from strict isotropic best records')
    parser.add_argument('--isotropic-run');parser.add_argument('--headless-train',action='store_true')
    parser.add_argument('--epochs',type=int,default=100)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.headless_train and not args.isotropic_run:parser.error('--isotropic-run is required; no parameter defaults are substituted')
    c=Controller();c.settings.update(epochs=args.epochs);gui_initialized=False
    try:
        if args.isotropic_run:c.path=args.isotropic_run;c.load()
        if args.headless_train:
            c.start()
            while c.running or c.child is not None:c.tick();time.sleep(.1)
            if c.error:raise RuntimeError(c.error)
        else:
            ps.init();gui_initialized=True;vertices=[];faces=[]
            for line in (ROOT/'mesh.obj').read_text().splitlines():
                tokens=line.split()
                if tokens and tokens[0]=='v':vertices.append([float(x) for x in tokens[1:4]])
                if tokens and tokens[0]=='f':faces.append([int(x.split('/')[0])-1 for x in tokens[1:4]])
            c.mesh_handle=ps.register_surface_mesh('Joint bending latest directional mesh',np.array(vertices),np.array(faces))
            def callback():c.draw();c.tick()
            ps.set_user_callback(callback);ps.show()
    finally:
        c.close()
        if gui_initialized:set_logging_theme(True)

if __name__=='__main__':main()
