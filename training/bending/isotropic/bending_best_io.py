"""Durable, strict isotropic-to-orthotropic best-parameter handoff."""
import datetime,hashlib,io,json,math,os,time,uuid
from pathlib import Path
import numpy as np

ORDER=('kb','friction','thres','hardening','yield')
SCHEMA='isotropic-bending-handoff-v2'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_bytes(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with tmp.open('wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        for attempt in range(20):
            try:os.replace(tmp,path);return
            except PermissionError:
                if attempt==19:raise
                time.sleep(.02*(attempt+1))
    finally:
        if tmp.exists():tmp.unlink()
def atomic_json(path,value):atomic_bytes(path,json.dumps(value,indent=2,allow_nan=False).encode('utf-8'))


class BestStore:
    def __init__(self,root,settings):
        if type(settings.get('plastic')) is not bool:raise ValueError('Explicit plasticity setting required in isotropic settings')
        root=Path(root).resolve();self.settings=json.loads(json.dumps(settings));self.run_id=uuid.uuid4().hex
        self.directory=root/'logs'/('gui_bending_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.directory.mkdir(parents=True,exist_ok=False)
        self.manifest=dict(schema=SCHEMA,run_id=self.run_id,parameter_order=ORDER,lifecycle='running',plasticity_enabled=settings['plastic'],
            best_parameters_always_saved=True,detailed_records_optional=True,settings=self.settings,
            mesh=dict(path=str(root/'mesh.obj'),sha256=digest(root/'mesh.obj')),
            data={axis:dict(path=str(Path(settings[axis+'_data']).resolve()),sha256=digest(settings[axis+'_data'])) for axis in ('warp','weft')},
            stage_bests={})
        self.flush()
    def flush(self):atomic_json(self.directory/'run_manifest.json',self.manifest)
    def mark(self,state):
        self.manifest['lifecycle']=state;self.flush()
    def consider(self,stage,epoch,parameters,loss,pred,gt,x):
        parameters=np.asarray(parameters,dtype=float);pred=np.asarray(pred,dtype=float);gt=np.asarray(gt,dtype=float)
        if parameters.shape!=(5,) or not np.isfinite(parameters).all() or np.any(parameters<0):raise ValueError('Invalid isotropic best parameters')
        if not self.manifest['plasticity_enabled'] and not np.array_equal(parameters[3:],[50.,100.]):raise ValueError('No-plastic records require fixed hardening=50 and yield=100')
        if not math.isfinite(loss) or not np.isfinite(pred).all() or not np.isfinite(gt).all() or pred.shape!=gt.shape:raise ValueError('Invalid best loss/curve')
        selected=list(range(30,71)) if stage==1 else list(range(30,71))+list(range(120,151))
        if pred.shape!=(len(selected),):raise ValueError('Best curve has incorrect selected-frame count')
        if not math.isclose(float(np.mean((pred-gt)**2)),float(loss),rel_tol=1e-6,abs_tol=1e-12):raise ValueError('Best curve/loss mismatch')
        old=self.manifest['stage_bests'].get(str(stage))
        if old and loss>=old['loss']:return
        axis='weft' if stage==3 else 'warp';revision=f'stage{stage}_epoch{epoch:05d}_{uuid.uuid4().hex}'
        curve=self.directory/(revision+'.npz');buffer=io.BytesIO()
        np.savez_compressed(buffer,parameters=parameters,pred=pred,gt=gt,displacement_mm=np.asarray(x),loss=loss,epoch=epoch,stage=stage)
        atomic_bytes(curve,buffer.getvalue())
        payload=dict(schema=SCHEMA,run_id=self.run_id,stage=stage,direction=axis,epoch=epoch,loss=float(loss),
            parameter_order=ORDER,parameters=parameters.tolist(),frames=selected,plasticity_enabled=self.manifest['plasticity_enabled'],forward_evaluated=True,
            gradient_completion_not_required=True,curve_file=curve.name,curve_sha256=digest(curve))
        record=self.directory/(revision+'.json');atomic_json(record,payload)
        self.manifest['stage_bests'][str(stage)]=dict(file=record.name,sha256=digest(record),loss=float(loss),epoch=epoch,direction=axis,plasticity_enabled=self.manifest['plasticity_enabled'])
        self.flush()


def load_isotropic_run(folder,mesh=None):
    folder=Path(folder).expanduser().resolve(strict=True)
    manifest_file=folder/'run_manifest.json'
    if not manifest_file.is_file():raise FileNotFoundError('Missing run_manifest.json. Select an isotropic run created with the durable-best version; no default parameters will be substituted.')
    manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest.get('schema')!=SCHEMA or manifest.get('parameter_order')!=list(ORDER):raise ValueError('Unsupported isotropic record schema/parameter order')
    if type(manifest.get('plasticity_enabled')) is not bool:raise ValueError('Isotropic log has no explicit plasticity setting; no default is inferred')
    if manifest.get('settings',{}).get('plastic') is not manifest['plasticity_enabled']:raise ValueError('Manifest and settings plasticity mismatch')
    required=('fabric','density','thickness','warp','weft','material')
    if not all(k in manifest.get('settings',{}) for k in required):raise ValueError('Missing material configuration in isotropic run')
    for axis in ('warp','weft'):
        info=manifest['data'][axis]
        if digest(info['path'])!=info['sha256']:raise ValueError(axis+' input data have changed since isotropic training')
    if digest(manifest['mesh']['path'])!=manifest['mesh']['sha256']:raise ValueError('Isotropic mesh has changed')
    if mesh is not None and digest(mesh)!=manifest['mesh']['sha256']:raise ValueError('Joint mesh does not match the isotropic training mesh')
    bests={}
    for stage,axis in [(2,'warp'),(3,'weft')]:
        entry=manifest.get('stage_bests',{}).get(str(stage))
        if entry is None:raise FileNotFoundError(f'Missing {axis} five-parameter best (stage {stage}). The run may have been interrupted before this direction was evaluated.')
        file=(folder/entry['file']).resolve()
        if not file.is_relative_to(folder) or digest(file)!=entry['sha256']:raise ValueError('Best parameter file failed path/hash verification')
        payload=json.loads(file.read_text(encoding='utf-8'))
        if (payload.get('schema')!=SCHEMA or payload.get('run_id')!=manifest['run_id'] or payload.get('stage')!=stage or
                payload.get('direction')!=axis or payload.get('parameter_order')!=list(ORDER)):
            raise ValueError('Direction, stage, run ID or parameter order mismatch')
        if payload.get('plasticity_enabled') is not manifest['plasticity_enabled'] or entry.get('plasticity_enabled') is not manifest['plasticity_enabled']:raise ValueError('Warp/weft plasticity settings do not match the run')
        values=np.asarray(payload['parameters'],dtype=float)
        if values.shape!=(5,) or not np.isfinite(values).all() or np.any(values<0) or not math.isfinite(payload['loss']):raise ValueError('Invalid best parameter values')
        if not manifest['plasticity_enabled'] and not np.array_equal(values[3:],[50.,100.]):raise ValueError('No-plastic best has inconsistent fixed plastic parameters')
        curve=(folder/payload['curve_file']).resolve()
        if not curve.is_relative_to(folder) or digest(curve)!=payload['curve_sha256']:raise ValueError('Best curve failed hash verification')
        with np.load(curve,allow_pickle=False) as arrays:
            if not np.array_equal(arrays['parameters'],values):raise ValueError('Best JSON and curve parameters disagree')
            if not math.isclose(float(np.mean((arrays['pred']-arrays['gt'])**2)),payload['loss'],rel_tol=1e-6,abs_tol=1e-12):raise ValueError('Best curve MSE disagrees with saved loss')
            selected=list(range(30,71))+list(range(120,151))
            if payload.get('frames')!=selected:raise ValueError('Unexpected isotropic loss-frame selection')
            data=np.load(manifest['data'][axis]['path'],allow_pickle=False)
            if not np.array_equal(arrays['gt'],data[1,selected]) or not np.array_equal(arrays['displacement_mm'],data[0,selected]):
                raise ValueError('Saved GT/displacement does not match the recorded input data and frames')
        bests[axis]=values.tolist()
    return dict(directory=str(folder),manifest=manifest,bests=bests,plasticity_enabled=manifest['plasticity_enabled'])
