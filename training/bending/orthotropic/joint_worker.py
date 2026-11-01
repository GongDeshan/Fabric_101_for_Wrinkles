"""One cold-process directional rollout, using the established history-off model."""
import ast,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODEL_NAME='LUFHES_elastic_stvk_lr_compress_orthotropic_normalized_training_copy.py'

def worker(request,response):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    req=json.loads(request.read_text());out=request.parent
    sys.path.insert(0,str(out/'snapshot'))
    from train_remaining_bending_pipeline import assign
    source=(out/'snapshot'/MODEL_NAME).read_text(encoding='utf-8-sig')
    source=source.rsplit('\nif _cli_args.audit_position_frames is not None:\n',1)[0]
    def once(a,b):
        nonlocal source
        assert source.count(a)==1,(a,source.count(a));source=source.replace(a,b)
    once('_log_dir = os.path.join(os.path.dirname(__file__), "logs")','_log_dir = '+repr(str(out/'logs')))
    once('P = self._lower + (self._upper - self._lower) * torch.sigmoid(self.U())','P = self.U().square()')
    source=assign(source,dict(MESH_PATH=repr(str(ROOT/'mesh.obj')),
        _BENDING_PARAMETER_LOWER=repr(tuple([-1.]*10)),_BENDING_PARAMETER_UPPER=repr(tuple([1e30]*10))))
    fixed=req['settings'];material=fixed['material']
    source=assign(source,{k:repr(v) for k,v in dict(Warp_Stiffness=fixed['warp'],Weft_Stiffness=fixed['weft'],
        MASS_DENSITY=fixed['density'],THICKNESS=fixed['thickness'],K33_0=material['k33_0'],K33_1=material['k33_1'],
        Fric_K33=material['friction'],Hardening_K33=material['hardening'],Thres_uv=100.,Yield_UV=100.).items()})
    data_lines=[line for line in source.splitlines() if 'with open(f' in line and 'compression.npy' in line]
    assert len(data_lines)==1
    once(data_lines[0],'with open('+repr(req['data_path'])+', "rb") as f:')
    once('gt_comp_force_all = torch.from_numpy','_training_frames = np.r_[30:71,120:151].astype(np.int32)\ngt_comp_force_all = torch.from_numpy')
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='MyDiffSimParameter')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='set_physical')
    lines=source.splitlines(keepends=True)
    lines[method.lineno-1:method.end_lineno]=['''    def set_physical(self, values):
        values=torch.as_tensor(values,dtype=torch.float64)
        if not torch.isfinite(values).all() or not (values>=0).all():raise ValueError('Invalid physical parameters')
        self.U().copy_(values.sqrt())
''']
    source=''.join(lines)
    (out/'executed_scene.py').write_text(source,encoding='utf-8')
    sys.argv=[str(ROOT/MODEL_NAME),'--headless-train','--fabric',req['fabric'],'--stage',req['direction'],
        '--mechanism-phase','refine','--bending-history-autograd','--training-end-frame','150',
        '--loss-frame-range','30','151','--solver-tolerances',str(fixed['newton']),str(fixed['linear']),
        *[item for name in req['active'] for item in ('--active-parameter',name)],'--epochs','1']
    if fixed['contact']:sys.argv.append('--training-contact')
    for name,value in req['parameters'].items():sys.argv+=['--initial-parameter',f'{name}={value:.17g}']
    ns=dict(__name__='joint_gui_directional_worker',__file__=str(ROOT/MODEL_NAME))
    exec(compile(source,str(out/'executed_scene.py'),'exec'),ns)
    assert ns['config']['diff_sim']['orthotropic_bending_history_gradient'] is False
    assert bool(ns['config']['contact']['enable'])==fixed['contact']
    assert ns['tested_fabric_type']==req['fabric'] and tuple(ns['_active_parameter_names'])==tuple(req['active'])
    names=tuple(ns['_BENDING_PARAMETER_NAMES'])
    signed=torch.tensor(req['signed_U'],dtype=torch.double)
    np.testing.assert_allclose(signed.square().numpy(),[req['parameters'][n] for n in names],rtol=1e-12,atol=1e-12)
    with torch.no_grad():ns['parm'].U()[:10].copy_(signed);ns['parm'].sync()
    loss=ns['loss_fn'](training=req['gradient'],verbose=False)
    assert torch.equal(ns['parm'].U()[:10].detach().cpu(),signed),'Rollout changed signed U'
    assert np.array_equal(ns['_training_frames'],np.r_[30:71,120:151])
    module=ns['diff_module'];plastic={}
    for axis in ('warp','weft'):
        values=np.asarray(getattr(module,'get_plastic_thetas_'+axis)());rest=getattr(module,'_rest_plastic_thetas_'+axis)
        plastic[axis]=float(np.max(np.abs(values-rest)))
        if not fixed['plastic']:assert plastic[axis]<1e-12,plastic
    result=dict(loss=float(loss.detach()),parameters=req['parameters'],signed_U=req['signed_U'],prediction=ns['last_prediction'].tolist(),
        gt=ns['gt_comp_force'].detach().cpu().tolist(),frames=ns['_training_frames'].tolist(),plastic_max_change=plastic,history_gradient=False)
    if req['gradient']:result['gradient_U']=ns['parm'].U().grad.detach().cpu().numpy()[:10].tolist()
    assert np.isfinite(result['loss']) and np.isfinite(result['prediction']).all()
    if req['gradient']:assert np.isfinite(result['gradient_U']).all()
    x=module._X.detach().cpu().numpy().reshape(150,ns['NUM_VERTICES'],3)[-1]
    np.savez_compressed(out/'last_mesh.npz',vertices=x)
    tmp=response.with_suffix('.tmp.json');tmp.write_text(json.dumps(result,allow_nan=False),encoding='utf-8');tmp.replace(response)

if __name__=='__main__':
    if sys.stdin.readline().strip()!='START':raise RuntimeError('Parent did not authorize worker start')
    worker(Path(sys.argv[1]),Path(sys.argv[2]))
