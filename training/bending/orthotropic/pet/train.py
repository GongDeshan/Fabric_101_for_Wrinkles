"""Run the PET joint fit with fixed weft plastic parameters."""
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from windows_training_lifetime_copy import Lifetime
from pet_joint_worker import prepare_source

ROOT = Path(__file__).resolve().parent
NAMES = ('kb_warp', 'kb_weft', 'friction_warp', 'friction_weft',
         'hardening_warp', 'hardening_weft', 'yield_warp', 'yield_weft', 'thres_warp', 'thres_weft')
INDICES = [0, 1, 2, 3, 4, 6, 8, 9]
ACTIVE = tuple(NAMES[i] for i in INDICES)
RATES = [.3, .3, .3, .3, .1, .001, .002, .002]


def save(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def load_configuration():
    protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    paths = {
        '62_pet_warp_compression.npy': ROOT / '62_pet_warp_compression.npy',
        '62_pet_weft_compression.npy': ROOT.parent.parent / 'isotropic' / 'data' / '62_pet_weft_compression.npy',
        'mesh.obj': ROOT.parent / 'mesh.obj',
    }
    for name, path in paths.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != protocol['inputs_sha256'][name]:
            raise ValueError(f'Input does not match the PET experiment: {path}')
    initial = {}
    for direction in ('warp', 'weft'):
        record = json.loads((ROOT / f'{direction}_isotropic_best.json').read_text(encoding='utf-8'))
        initial.update((f'{name}_{direction}', value) for name, value in record['parameters'].items())
    if not math.isclose(initial['hardening_weft'], 50, rel_tol=1e-15):
        raise ValueError('The recorded weft hardening must be 50')
    initial['yield_weft'] = 3.14
    if set(initial) != set(NAMES) or not all(math.isfinite(v) and v > 0 for v in initial.values()):
        raise ValueError('The PET initialization must specify ten finite, positive parameters')
    if tuple(protocol['active']) != ACTIVE or protocol['learning_rates_U'] != RATES:
        raise ValueError('The PET optimizer does not match the recorded protocol')
    return protocol, initial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--check-only', action='store_true', help='Verify inputs and compile the worker source without a rollout')
    parser.add_argument('--output', type=Path, help='New output folder (default: runs/<timestamp>)')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    protocol, initial = load_configuration()
    if args.check_only:
        for direction in ('warp', 'weft'):
            compile(prepare_source(direction, ROOT), f'<PET {direction} worker>', 'exec')
        print(json.dumps(dict(initial=initial, active=ACTIVE, fixed=protocol['fixed']), indent=2))
        return
    out = args.output.resolve() if args.output else ROOT / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    lifetime = Lifetime(out / 'training.lock')
    state = dict(status='initializing', pid=os.getpid())
    def status(**kwargs):
        state.update(kwargs); save(out / 'status.json', state)
    status()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        save(out / 'protocol.json', dict(protocol, epochs=args.epochs, initial=initial))
        u = torch.nn.Parameter(torch.tensor([initial[n] for n in ACTIVE], dtype=torch.float64).sqrt())
        optimizer = torch.optim.Adam([u], lr=1.)
        environment = os.environ.copy()
        for key in ('UIPC_NEWTON_AUDIT', 'UIPC_FRAME_TRACE'):
            environment.pop(key, None)
        start = time.perf_counter()
        def physical():
            p = dict(initial); p.update(zip(ACTIVE, u.detach().square().tolist())); return p
        def evaluate(p, label, gradient):
            full_u = [p[n] ** .5 for n in NAMES]
            for index, value in zip(INDICES, u.detach().tolist()):
                full_u[index] = value
            results = []
            for direction in ('warp', 'weft'):
                if (out / 'STOP').exists():
                    raise KeyboardInterrupt('STOP requested')
                folder = out / f'{label}_{direction}'
                folder.mkdir()
                request = folder / 'request.json'; response = folder / 'result.json'
                save(request, dict(direction=direction, parameters=p, signed_U=full_u, gradient=gradient))
                status(status='running', evaluation=label, direction=direction)
                with (folder / 'worker.log').open('w', encoding='utf-8') as log:
                    child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'pet_joint_worker.py'), str(request), str(response)],
                        cwd=ROOT, env=environment, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                        text=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    try:
                        lifetime.assign(child)
                        status(child_pid=child.pid)
                        child.stdin.write('START\n'); child.stdin.flush()
                        deadline = time.monotonic() + 3600
                        while child.poll() is None:
                            if (out / 'STOP').exists(): raise KeyboardInterrupt('STOP requested')
                            if time.monotonic() > deadline: raise TimeoutError('Evaluation exceeded one hour')
                            time.sleep(.25)
                        if child.returncode: raise RuntimeError(f'{label}/{direction} failed: {folder / "worker.log"}')
                    finally:
                        if child.poll() is None: child.kill()
                        child.wait(timeout=10); child.stdin.close()
                        status(child_pid=None)
                result = json.loads(response.read_text())
                assert result['signed_U'] == full_u and result['parameters'] == p
                results.append(result)
            return results
        best = float('inf'); reference = float('inf'); stale = 0
        best_p = physical()
        for epoch in range(args.epochs):
            before = time.perf_counter()
            p = physical(); results = evaluate(p, f'epoch_{epoch:03}', True)
            loss = sum(r['loss'] for r in results) / 2
            if not np.isfinite(loss): raise RuntimeError('Nonfinite loss')
            if loss < best:
                best = loss; best_p = dict(p)
                save(out / 'best.json', dict(epoch=epoch, loss=loss, parameters=p, results=results))
            if epoch == 0:
                save(out / 'initial_evaluation.json', dict(loss=loss, parameters=p, results=results))
            if loss < reference * .999:
                reference = loss; stale = 0
            else: stale += 1
            gradients = torch.tensor([r['gradient_U'] for r in results], dtype=torch.float64)
            assert gradients.shape == (2, 10) and torch.isfinite(gradients).all()
            gradient = gradients.mean(0)[INDICES]
            optimizer.zero_grad(set_to_none=True); u.grad = gradient
            old = u.detach().clone(); optimizer.step()
            with torch.no_grad(): u.copy_(old + torch.tensor(RATES) * (u - old))
            row = dict(epoch=epoch, loss=loss, warp_loss=results[0]['loss'], weft_loss=results[1]['loss'],
                       parameters=p, after_step=physical(), gradient_U=gradient.tolist(), best_loss=best,
                       plastic_check={d:r['plastic_check'] for d,r in zip(('warp','weft'),results)},
                       stale=stale, seconds=time.perf_counter() - before, elapsed=time.perf_counter() - start)
            with (out / 'epochs.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row) + '\n')
            torch.save(dict(epoch=epoch, u=u.detach(), optimizer=optimizer.state_dict(), best=best,
                            best_parameters=best_p, active=ACTIVE), out / 'last_checkpoint.pt')
            status(epoch=epoch, latest=row, best_loss=best, best_parameters=best_p)
            print('JOINT_EPOCH', json.dumps(row), flush=True)
            if epoch + 1 >= 30 and stale >= 20: break
        with torch.no_grad():
            u.copy_(torch.tensor([best_p[n] for n in ACTIVE], dtype=torch.float64).sqrt())
        status(status='best_replay')
        replay = evaluate(best_p, 'best_replay', False)
        replay_loss = sum(r['loss'] for r in replay) / 2
        save(out / 'summary.json', dict(best_loss=best, best_parameters=best_p, replay_loss=replay_loss,
                                       replay=replay, epochs=epoch + 1))
        status(status='complete', best_loss=best, best_replay_loss=replay_loss, best_parameters=best_p,
               epochs=epoch + 1, child_pid=None)
    except BaseException as exc:
        status(status='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', error=traceback.format_exc())
        raise
    finally:
        lifetime.close()


if __name__ == '__main__':
    main()
