"""Incremental training records for the stretch GUI (no extra simulations)."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import numpy as np


class StretchRecordWriter:
    def __init__(self, env):
        self.env = env
        self.directory = None
        self.best = float('inf')
        self.count = 0

    def _json(self, path, data):
        tmp = path.with_suffix('.tmp.json')
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(tmp, path)

    def _npz(self, path, **values):
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(tmp, **values)
        os.replace(tmp, path)

    def save(self, epoch, pred, stiffness, loss, seconds, next_stiffness):
        e = self.env
        if self.directory is None:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.directory = Path(e['__file__']).resolve().parent/'logs'/('gui_training_'+stamp)
            self.directory.mkdir(parents=True, exist_ok=False)
            manifest = {k:e[k] for k in ['GT_PATH','MASS_DENSITY','THICKNESS','K11','K22','PARAM_LR','NUM_EPOCH','EndFrame','sample_vertial_direction']}
            manifest.update(parameterization='P=U^2', units=dict(stiffness='Pa',density='kg/m^3',thickness='m',force='N',displacement='mm'),
                epoch_numbering='one-based', save_policy='Only epochs completed while saving is enabled are written; disabled intervals remain gaps.',
                curve_parameter='stiffness_evaluated (before optimizer update)',
                data_sha256=hashlib.sha256(Path(e['GT_PATH']).read_bytes()).hexdigest(),
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(e['__file__']).parent.glob('*.py')})
            self._json(self.directory/'config.json', manifest)
        gt = e['forces_gt'].detach().cpu().numpy().copy()
        x = np.asarray(e['real_load_pos_forces'][0]).copy()
        values = dict(epoch=int(epoch), pred=np.asarray(pred).copy(), gt=gt, displacement_mm=x,
                      K22=float(stiffness), loss=float(loss), seconds=float(seconds))
        self._npz(self.directory/f'epoch_{epoch:05d}.npz', **values)
        row = dict(epoch=int(epoch), loss=float(loss), stiffness_evaluated=float(stiffness),
                   stiffness_after_update=float(next_stiffness), seconds=float(seconds),
                   curve_file=f'epoch_{epoch:05d}.npz')
        with (self.directory/'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.count += 1
        if loss < self.best:
            self._npz(self.directory/'best_result.npz', **values)
            self.best = float(loss)
        self._json(self.directory/'summary.json', dict(saved_epochs=self.count, last_saved_epoch=epoch,
                    best_saved_loss=self.best, note='Best among saved epochs only. Records do not contain Adam state.'))
