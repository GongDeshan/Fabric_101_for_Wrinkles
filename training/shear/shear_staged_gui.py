"""Interactive two-stage shear training; keeps the supplied constitutive readout."""
import ast
import contextlib
import datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import polyscope as ps
from polyscope import imgui
from stretch_logging_theme import set_logging_theme
from shear_torque_plot import draw_torque_overlay

NAMES = ('k33_0', 'k33_1', 'friction_k33', 'thres_uv', 'yield_uv', 'hardening_k33')
BOUNDS = ((1e3, 1.5e6), (1e3, 1.5e6), (1e3, 2e6),
          (1e-4, .33*.029), (.33*.029, .98*.029), (1e3, 1.5e6))


def validate(settings):
    data = np.load(settings['target'], allow_pickle=False)
    if data.ndim != 2 or data.shape[0] != 2 or data.shape[1] < 61 or not np.isfinite(data).all():
        raise ValueError('Expected finite (2, N), N >= 61: angle in degrees, torque in N m.')
    expected = np.concatenate((np.linspace(0, 5, 21), np.linspace(5, 0, 21)[1:], np.linspace(0, 5, 21)[1:]))
    if not np.allclose(data[0, :61], expected, atol=1e-4, rtol=1e-4):
        raise ValueError('First 61 angles must match the 0 -> 5 -> 0 -> 5 degree schedule (20 steps per branch). No resampling is performed.')
    values = [settings[k] for k in ('density', 'thickness', 'warp', 'weft', 'lr')]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('Density, thickness, fixed stretch stiffnesses and learning rate must be positive and finite.')
    for name, value, (lo, hi) in zip(NAMES, settings['parameters'], BOUNDS):
        if not math.isfinite(value) or not lo < value < hi:
            raise ValueError(f'{name} must be strictly inside ({lo:g}, {hi:g}).')
    if min(settings['epochs1'], settings['epochs2']) < 1:
        raise ValueError('Both stage epoch limits must be positive.')
    return data


def prepare_source(source, settings, stage):
    """Apply stage mapping before world.init, as in the verified staged runs."""
    count, end = (4, 40) if stage == 1 else (6, 60)
    source = source[:source.index('\nworld.dump()\n')] + '\nworld.dump()\n'
    a = source.index('class _Tee:')
    b = source.index('FABRIC_INITIAL_GUESSES =', a)
    source = source[:a] + "_log_path = str(Path(__file__).resolve().parent/'logs/gui_shear')\n" + source[b:]
    source = 'from pathlib import Path\n' + source
    material = dict(warp=settings['warp'], weft=settings['weft'], density=settings['density'], thickness=settings['thickness'],
                    **dict(zip(NAMES, settings['parameters'])))
    tree = ast.parse(source)
    replacements = dict(tested_fabric_type=settings['fabric'], Up_Direction=settings['direction'],
        EndFrame=end, PARAM_LR=settings['lr'], NUM_EPOCH=settings['epochs1' if stage == 1 else 'epochs2'],
        training_data_path=str(Path(settings['target']).resolve()), IS_SAVE_LOAD_DEFORMATION=False,
        _material=material)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in replacements:
                node.value = ast.parse(repr(replacements[name]), mode='eval').body
    tree.body = [n for n in tree.body if not (isinstance(n, ast.If) and ast.unparse(n.test) in
        ('_cli_args.cotton_warm_start', '_cli_args.epochs is not None'))]
    class Quiet(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) in ('print', 'Timer.report'):
                return ast.copy_location(ast.Pass(), node)
            return self.generic_visit(node)
    tree = Quiet().visit(tree)
    source = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    if stage == 1:
        source = source.replace('enumerate(PARAM_NAMES):', 'enumerate(PARAM_NAMES[:grad.numel()]):')
        source = source.replace('class MyDiffSimParameter(DiffSimParameter):',
            'SHEAR_PARAMETER_BOUNDS = SHEAR_PARAMETER_BOUNDS[:4]\nclass MyDiffSimParameter(DiffSimParameter):')
        source = source.replace('scene=scene, size=6,', 'scene=scene, size=4,')
        source = source.replace('p_ref=[K33_0, K33_1, Fric_K33, Thres_uv, Yield_UV, Hardening_K33]',
            'p_ref=[K33_0, K33_1, Fric_K33, Thres_uv]')
        source = source.replace('    P = parm.forward()\n',
            '    P = torch.cat((parm.forward(), torch.tensor([Yield_UV, Hardening_K33], dtype=torch.float64)))\n')
        tree = ast.parse(source)
        def excluded(node):
            text = ast.unparse(node)
            return isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
                token in text for token in ('diff_yield_uv', 'diff_hardening_k33'))
        tree.body = [n for n in tree.body if not excluded(n)]
        source = ast.unparse(ast.fix_missing_locations(tree))+'\n'
    compile(source, '<configured shear scene>', 'exec')
    return source


class Records:
    def __init__(self, root):
        self.root = root
        self.directory = None
        self.best = {}
        self.counts = {}

    def save(self, row, settings, source):
        if self.directory is None:
            self.directory = self.root/'logs'/('gui_staged_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.directory.mkdir(parents=True, exist_ok=False)
        stage = row['stage']
        folder = self.directory/f'stage{stage}'
        folder.mkdir(exist_ok=True)
        config = folder/'config.json'
        if not config.exists():
            (folder/'scene_source.py').write_text(source,encoding='utf-8')
            config.write_text(json.dumps(dict(settings=settings, frames=[0, 40 if stage == 1 else 60],
                active_parameters=list(NAMES[:4 if stage == 1 else 6]), parameterization='bounded sigmoid',
                data_sha256=hashlib.sha256(Path(settings['target']).read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256(source.encode()).hexdigest(),
                saved_epochs_only=True, includes_adam_state=False), indent=2), encoding='utf-8')
        base = f'epoch_{row["epoch"]:05d}'
        values = {key:row[key] for key in ('pred', 'gt', 'angle', 'parameters', 'loss', 'stage', 'epoch', 'seconds')}
        np.savez_compressed(folder/(base+'.npz'), **values)
        scalar = {k:v for k,v in row.items() if k not in ('pred','gt','angle')}
        with (folder/'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(scalar, allow_nan=False)+'\n');stream.flush()
        if row['loss'] < self.best.get(stage, float('inf')):
            np.savez_compressed(folder/'best_result.npz', **values)
            self.best[stage] = row['loss']
        self.counts[stage]=self.counts.get(stage,0)+1
        (folder/'summary.json').write_text(json.dumps(dict(saved_epochs=self.counts[stage],
            last_saved_epoch=row['epoch'],best_saved_loss=self.best[stage],best_among_saved_epochs_only=True),indent=2),encoding='utf-8')


class Controller:
    def __init__(self, root, settings):
        self.root = root
        self.settings = settings
        self.stage = 1
        self.records = Records(root)
        self.save_records = True
        self.save_error = ''
        self.history = []
        self.follow = True
        self.curve_index = 0
        self.log_axis = False
        self.message = 'Ready'
        self.error = ''
        self.pending = None
        self.running = False
        self.finished = False
        self.env = None

    def attach(self, env, source):
        self.env, self.scene_source = env, source
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_parameters = list(self.settings['parameters'])
        self.reference = float('inf')
        self.stale = 0
        self.stage_losses = []
        self.stage_times = []
        self.capture = None
        def capture(loss, P, torques):
            self.capture = dict(loss=float(loss.detach()), parameters=P.detach().cpu().tolist(),
                                pred=torques.detach().cpu().numpy().copy())
        env['save_best_checkpoint'] = capture

    def start(self):
        if self.finished:
            self.message = 'Finished. Apply settings to start a new two-stage run.'
            return
        self.env['run'] = False
        self.running = True
        self.defer = True
        self.message = 'Training'

    def tick(self):
        if not self.running:return
        if self.defer:
            self.defer = False;return
        e = self.env
        try:
            self.capture = None
            e['_current_epoch'] = self.epoch
            started = time.perf_counter()
            with contextlib.redirect_stdout(__import__('io').StringIO()):
                value = float(e['optimizer'].step(e['loss_fn']).detach())
            seconds = time.perf_counter()-started
            assert self.capture is not None, 'Loss did not provide a prediction'
            row = dict(self.capture)
            gt = e['torque_gt'].detach().cpu().numpy().copy()
            assert np.isfinite(value) and np.isfinite(row['pred']).all()
            assert math.isclose(float(np.mean((row['pred']-gt)**2)), value, rel_tol=1e-6, abs_tol=1e-12)
            after = e['parm'].forward().detach().cpu().tolist()
            if self.stage == 1:after += self.settings['parameters'][4:]
            self.epoch += 1
            row.update(stage=self.stage, epoch=self.epoch, seconds=seconds, gt=gt,
                angle=e['raw_twist_angles'][:len(gt)].copy(), after_step=after,
                gradient_U=e['parm'].U().grad.detach().cpu().tolist())
            assert len(row['parameters']) == 6
            self.history.append(row);self.stage_losses.append(value);self.stage_times.append(seconds)
            if value < self.best_loss:
                self.best_loss = value;self.best_parameters = list(row['parameters'])
            median = float(np.median(self.stage_losses[-5:]))
            if median < self.reference*.999:self.reference=median;self.stale=0
            else:self.stale+=1
            if self.save_records:
                try:
                    self.records.save(row,self.settings,self.scene_source);self.save_error=''
                except Exception as exc:
                    self.save_error='Saving failed: '+str(exc);self.save_records=False
                    traceback.print_exc()
            maximum = self.settings['epochs1' if self.stage == 1 else 'epochs2']
            plateau = self.settings['early_stop'] and self.epoch>=30 and self.stale>=20
            if self.epoch >= maximum or plateau:
                self.message = 'Plateau early stop' if plateau else 'Epoch limit reached'
                self.running = False
                if self.stage == 1:
                    self.settings['parameters'] = list(self.best_parameters)
                    self.stage = 2;self.pending='next_stage'
                else:
                    self.finished = True
                    with torch.no_grad():
                        for i,(lo,hi) in enumerate(BOUNDS):
                            p=self.best_parameters[i];e['parm'].U()[i]=math.log((p-lo)/(hi-p))
                        e['parm'].sync()
                        e['lr']._old_U.copy_(e['parm'].U())
                        e['lr']._old_P.copy_(e['parm'].forward())
                    self.message += '; stage 2 best parameters restored (not replayed)'
        except Exception as exc:
            self.running=False;self.error=str(exc);self.message='Training failed';traceback.print_exc()

    def draw(self):
        e=self.env
        imgui.Text(f'Shear training: stage {self.stage}/2 ({4 if self.stage == 1 else 6} parameters)')
        _,self.save_records=imgui.Checkbox('Save training records',self.save_records)
        set_logging_theme(self.save_records)
        if not self.save_records:imgui.Text('LOG SAVING IS OFF')
        if self.records.directory:imgui.Text('Records: '+str(self.records.directory))
        if self.save_error:imgui.Text(self.save_error)
        maximum=self.settings['epochs1' if self.stage==1 else 'epochs2']
        imgui.Text(f'Epoch: {self.epoch}/{maximum}; best stage MSE: {self.best_loss:.9g}')
        imgui.Text(self.message)
        if self.error:imgui.Text(self.error)
        if self.stage_times:
            avg=float(np.mean(self.stage_times))
            imgui.Text(f'Last: {self.stage_times[-1]:.2f} s/epoch; mean: {avg:.2f} s/epoch')
            imgui.Text(f'Remaining current-stage compute estimate: {max(0,maximum-self.epoch)*avg/60:.1f} min (early stop may shorten it)')
        current=e['parm'].forward().detach().cpu().tolist()
        if len(current)==4:current+=self.settings['parameters'][4:]
        for name,value in zip(NAMES,current):imgui.Text(f'{name}: {value:.9g}')
        _,self.log_axis=imgui.Checkbox('Log10 loss axes',self.log_axis)
        for stage in (1,2):
            values=np.array([r['loss'] for r in self.history if r['stage']==stage],dtype=np.float64)
            if len(values):
                if self.log_axis:values=np.log10(np.maximum(values,1e-300))
                imgui.PlotLines(f'##stage{stage}_loss',values.astype(np.float32),overlay_text=f'Stage {stage}: loss vs epoch',graph_size=(300.,130.))
                imgui.Text(f'Stage {stage}: epoch 1 to {len(values)}; frame range 0-{40 if stage==1 else 60}')
        if self.history:
            _,self.follow=imgui.Checkbox('Show latest epoch torque',self.follow)
            if self.follow:self.curve_index=len(self.history)
            else:
                _,self.curve_index=imgui.InputInt('Curve record',self.curve_index)
                self.curve_index=max(1,min(len(self.history),self.curve_index))
            row=self.history[self.curve_index-1]
            imgui.Text(f'Torque curve: stage {row["stage"]}, epoch {row["epoch"]}, MSE={row["loss"]:.9g}')
            draw_torque_overlay(row['angle'],row['gt'],row['pred'])
        if self.running and imgui.Button('Pause optimization'):
            self.running=False;self.message='Paused'
        imgui.Text('The GUI refreshes between epochs. Epoch computation blocks the window until completion.')


class SettingsPanel:
    def __init__(self, controller, presets):
        self.controller=controller
        self.values=json.loads(json.dumps(controller.settings))
        self.files=sorted((controller.root/'data').glob('*.npy'))
        self.index=next((i for i,p in enumerate(self.files) if str(p.resolve())==str(Path(self.values['target']).resolve())),len(self.files))
        self.presets=presets
        self.error=''

    def draw(self):
        v=self.values;c=self.controller
        changed,self.index=imgui.Combo('Dataset',self.index,[p.name for p in self.files]+['Custom path'])
        if changed and self.index<len(self.files):
            p=self.files[self.index];v['target']=str(p)
            fabric=p.stem.rsplit('_',2)[0];direction=p.stem.rsplit('_',2)[1]
            v.update(fabric=fabric,direction=direction)
            if fabric in self.presets:
                m=self.presets[fabric]
                v.update(density=m['density'],thickness=m['thickness'],warp=m['warp'],weft=m['weft'],parameters=[m[n] for n in NAMES])
        _,v['target']=imgui.InputText('Data file (.npy)',v['target'],max_str_len=4096)
        changed,direction=imgui.Combo('Direction',0 if v['direction']=='warp' else 1,['warp','weft'])
        v['direction']=['warp','weft'][direction]
        for key,label in [('density','Density (kg/m^3)'),('thickness','Thickness (m)'),('warp','Fixed warp stretch stiffness (Pa)'),('weft','Fixed weft stretch stiffness (Pa)'),('lr','Adam LR (sigmoid U)')]:
            _,v[key]=imgui.InputFloat(label,float(v[key]),format='%.8g')
        for i,name in enumerate(NAMES):
            _,v['parameters'][i]=imgui.InputFloat('Initial '+name,float(v['parameters'][i]),format='%.8g')
        _,v['epochs1']=imgui.InputInt('Stage 1 maximum epochs',v['epochs1'])
        _,v['epochs2']=imgui.InputInt('Stage 2 maximum epochs',v['epochs2'])
        _,v['early_stop']=imgui.Checkbox('Plateau early stop (min 30, patience 20)',v['early_stop'])
        imgui.Text('Stage 1: frames 0-40; k33_0, k33_1, friction, threshold. Stage 2: frames 0-60; all six.')
        if imgui.Button('Apply settings and reset both stages'):
            try:
                validate(v);c.pending=('reset',json.loads(json.dumps(v)));c.running=False
            except Exception as exc:self.error=str(exc)
        if self.error:imgui.Text(self.error)


def run(model, args):
    torch.set_num_threads(4)
    model=Path(model).resolve();source=model.read_text(encoding='utf-8-sig')
    tree=ast.parse(source)
    preset_node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FABRIC_INITIAL_GUESSES' for t in n.targets))
    ns={'kPa':1000.};exec(compile(ast.Module(body=[preset_node],type_ignores=[]),'<presets>','exec'),ns)
    presets=ns['FABRIC_INITIAL_GUESSES'];fabric=args.fabric or '59_Cotton';m=presets[fabric]
    settings=dict(fabric=fabric,direction=args.direction,target=str(model.parent/'data'/f'{fabric}_{args.direction}_twist.npy'),
        density=m['density'],thickness=m['thickness'],warp=m['warp'],weft=m['weft'],lr=.02,
        parameters=[m[n] for n in NAMES],epochs1=args.epochs or 500,epochs2=args.epochs or 500,early_stop=True)
    if args.cotton_warm_start:
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='COTTON_WARM_START' for t in n.targets))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<warm start>','exec'),ns)
        settings['parameters']=[ns['COTTON_WARM_START'][n] for n in NAMES]
    validate(settings)
    controller=Controller(model.parent,settings)
    headless=args.headless_train
    if not headless:ps.init()
    dispatch=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=='FORWARD_VALIDATION' and n.orelse and any(isinstance(x,ast.If) and ast.unparse(x.test)=='_cli_args.headless_train' for x in n.orelse))
    gui_branch=dispatch.orelse[0].orelse[0]
    gui_body=[n for n in gui_branch.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func) in ('ps.init','ps.show'))]
    for node in gui_body:
        if isinstance(node,ast.FunctionDef) and node.name=='update':
            pos=next((i for i,n in enumerate(node.body) if not isinstance(n,ast.Global)),len(node.body))
            node.body[pos:pos]=ast.parse('''settings_panel.draw()
training_controller.draw()
if training_controller.pending:
    ps.unshow()
    return
''').body
            node.body.extend(ast.parse('''training_controller.tick()
if training_controller.pending:
    ps.unshow()
''').body)
            for statement in node.body:
                if isinstance(statement,ast.If) and ast.unparse(statement.test)=="imgui.Button('Run & Stop')":
                    statement.body[:0]=ast.parse("training_controller.running = False").body
            class PreviewParameters(ast.NodeTransformer):
                def visit_Assign(self, statement):
                    if (isinstance(statement.value,ast.Call) and ast.unparse(statement.value.func)=='net_force'
                            and any(isinstance(t,ast.Name) and t.id=='torque' for t in statement.targets)):
                        prefix=ast.parse('''_preview_P = parm.forward().detach()
if _preview_P.numel() == 4:
    _preview_P = torch.cat((_preview_P, torch.tensor([Yield_UV, Hardening_K33], dtype=torch.double)))
''').body
                        for argument,index in [(13,0),(14,1),(15,2),(16,3),(18,5),(19,4)]:
                            statement.value.args[argument]=ast.parse(f'_preview_P[{index}]',mode='eval').body
                        return prefix+[statement]
                    return self.generic_visit(statement)
            node=PreviewParameters().visit(node)
    class QuietGUI(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func) in ('print','Timer.report'):
                return ast.copy_location(ast.Pass(),node)
            return self.generic_visit(node)
    gui_body=[QuietGUI().visit(n) for n in gui_body]
    gui_code=compile(ast.fix_missing_locations(ast.Module(body=gui_body,type_ignores=[])),'<shear gui>','exec')
    while True:
        env=dict(__name__='_shear_managed_scene',__file__=str(model))
        scene=prepare_source(source,controller.settings,controller.stage)
        output=__import__('sys').stdout;error=__import__('sys').stderr
        try:
            with contextlib.redirect_stdout(__import__('io').StringIO()):
                exec(compile(scene,str(model),'exec'),env)
            controller.attach(env,scene)
            if controller.stage == 2:
                controller.start()
            panel=SettingsPanel(controller,presets)
            env.update(training_controller=controller,settings_panel=panel)
            env['optimize_once']=controller.start
            if headless:
                controller.start()
                while controller.running:controller.tick()
                if controller.error:raise RuntimeError(controller.error)
            else:
                env['sgui']=None
                exec(gui_code,env)
                ps.show()
        finally:
            if not headless:
                set_logging_theme(True);ps.set_user_callback(lambda:None);ps.remove_all_structures()
            controller.env=None
            env.clear();gc.collect()
        pending=controller.pending;controller.pending=None
        if pending=='next_stage':continue
        if isinstance(pending,tuple) and pending[0]=='reset':
            old_save=controller.save_records
            controller=Controller(model.parent,pending[1]);controller.save_records=old_save
            continue
        break
