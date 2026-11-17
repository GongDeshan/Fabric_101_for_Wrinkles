"""Ten-parameter full-model joint training, serial fresh workers, plasticity enabled."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from windows_training_lifetime_copy import Lifetime

ROOT = Path(__file__).resolve().parent
NAMES = ('kb_warp', 'kb_weft', 'friction_warp', 'friction_weft',
         'hardening_warp', 'hardening_weft', 'yield_warp', 'yield_weft', 'thres_warp', 'thres_weft')
INDICES = list(range(10))
ACTIVE = tuple(NAMES[i] for i in INDICES)
RATES = [.3, .3, .3, .3, 0., 0., 0., 0., .002, .002]


def save(path, value):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'training'
    out.mkdir(exist_ok=args.resume)
    lifetime = Lifetime(ROOT / 'training.lock')
    state = dict(status='initializing', pid=os.getpid())
    def status(**kwargs):
        state.update(kwargs); save(out / 'status.json', state)
    status()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        initial = json.loads((ROOT / 'initial.json').read_text(encoding='utf-8-sig'))['parameters']
        assert set(initial) == set(NAMES) and all(np.isfinite(v) and v > 0 for v in initial.values())
        files = [p for p in ROOT.iterdir() if p.suffix in ('.py', '.npy', '.npz', '.obj', '.json')]
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        save(out / ('resume_protocol.json' if args.resume else 'protocol.json'), dict(initial=initial, active=ACTIVE, learning_rates_U=RATES,
             frames=list(range(30, 71)) + list(range(120, 151)), forward_end_frame=150,
             solver=[.005, .0001], contact=True, soft_constraint=200,
             epochs=args.epochs, min_epochs=30, patience=20, relative_improvement=.001,
             parameterization='P=U^2; ten independent coordinates; fresh Adam; no extra LR hook',
             fresh_process_per_direction=True, serial_directions=True,
             initial_source='direction-specific full-model isotropic bests, from saved NPZ',
             plasticity='inactive; yield=100 and hardening=50 fixed in both directions',
             source_hashes=hashes))
        u = torch.nn.Parameter(torch.tensor([initial[n] for n in ACTIVE], dtype=torch.float64).sqrt())
        optimizer = torch.optim.Adam([u], lr=1.)
        environment = os.environ.copy()
        for key in ('UIPC_NEWTON_AUDIT', 'UIPC_FRAME_TRACE'):
            environment.pop(key, None)
        start = time.perf_counter()
        elapsed_offset = 0.
        def physical():
            p = dict(initial); p.update(zip(ACTIVE, u.detach().square().tolist())); return p
        def evaluate(p, label, gradient):
            full_u = [p[n] ** .5 for n in NAMES]
            for index, value in zip(INDICES, u.detach().tolist()):
                full_u[index] = value
            results = []
            for direction in ('warp', 'weft'):
                if (ROOT / 'STOP').exists():
                    raise KeyboardInterrupt('STOP requested')
                folder = out / f'{label}_{direction}'
                folder.mkdir()
                request = folder / 'request.json'; response = folder / 'result.json'
                save(request, dict(direction=direction, parameters=p, signed_U=full_u, gradient=gradient))
                status(status='running', evaluation=label, direction=direction)
                with (folder / 'worker.log').open('w', encoding='utf-8') as log:
                    child = subprocess.Popen([sys.executable, '-u', str(ROOT / 'full_joint_worker_copy.py'), str(request), str(response)],
                        cwd=ROOT, env=environment, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                        text=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    try:
                        lifetime.assign(child)
                        status(child_pid=child.pid)
                        child.stdin.write('START\n'); child.stdin.flush()
                        deadline = time.monotonic() + 3600
                        while child.poll() is None:
                            if (ROOT / 'STOP').exists(): raise KeyboardInterrupt('STOP requested')
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
        start_epoch = 0
        if args.resume:
            checkpoint = torch.load(out / 'last_checkpoint.pt', map_location='cpu', weights_only=False)
            assert tuple(checkpoint['active']) == ACTIVE
            with torch.no_grad(): u.copy_(checkpoint['u'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            saved = json.loads((out / 'best.json').read_text())
            best = saved['loss']; best_p = saved['parameters']
            rows = [json.loads(line) for line in (out / 'epochs.jsonl').read_text().splitlines() if line.strip()]
            assert rows[-1]['epoch'] + 1 == start_epoch
            elapsed_offset = rows[-1]['elapsed']
            for row in rows:
                if row['loss'] < reference * .999: reference = row['loss']; stale = 0
                else: stale += 1
            print('RESUMED', start_epoch, 'Adam state retained', flush=True)
        for epoch in range(start_epoch, args.epochs):
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
                       stale=stale, seconds=time.perf_counter() - before, elapsed=elapsed_offset + time.perf_counter() - start)
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
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
        save(out / 'summary.json', dict(best_loss=best, best_parameters=best_p, replay_loss=replay_loss,
                                       replay=replay, epochs=epoch + 1, originals_unchanged=True))
        status(status='complete', best_loss=best, best_replay_loss=replay_loss, best_parameters=best_p,
               epochs=epoch + 1, child_pid=None)
    except BaseException as exc:
        status(status='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', error=traceback.format_exc())
        raise
    finally:
        lifetime.close()


if __name__ == '__main__':
    assert sys.stdin.readline().strip() == 'START'
    main()
