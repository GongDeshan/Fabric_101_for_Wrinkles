"""Run the Wool shear and bending fits on Windows."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'common'))
from windows_training_lifetime_copy import Lifetime


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def verify_input(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Input differs from the reference experiment: {path}')


def prepare_stage(folder, components, mesh):
    folder.mkdir()
    for component in ('common', *components):
        for path in (ROOT / component).glob('*.py'):
            shutil.copyfile(path, folder / path.name)
    shutil.copyfile(mesh, folder / 'mesh.obj')


def run_stage(folder, filename, lifetime, epochs=None):
    log = folder / 'console.log'
    command = [sys.executable, '-X', 'utf8', '-u', str(folder / filename)]
    if epochs is not None:
        command += ['--epochs', str(epochs)]
    print(f'{folder.name}: {log}', flush=True)
    with log.open('w', encoding='utf-8') as stream:
        with subprocess.Popen(command, cwd=folder, stdin=subprocess.PIPE, stdout=stream,
                              stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW) as child:
            try:
                lifetime.assign(child)
                child.stdin.write(b'START\n')
                child.stdin.close()
                if child.wait():
                    raise RuntimeError(f'{folder.name} failed. See {log}')
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('shear', 'bending', 'joint'))
    parser.add_argument('--epochs', type=int, help='Override the original epoch limit for each stage')
    parser.add_argument('--output', type=Path, help='New output directory')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')

    protocol = load(ROOT / 'protocol.json')
    inputs = ROOT / 'inputs'
    meshes = {'shear': ROOT.parent / 'shear' / 'mesh.obj',
              'bending': ROOT.parent / 'bending' / 'orthotropic' / 'mesh.obj'}
    mesh_kind = 'shear' if args.mode == 'shear' else 'bending'
    mesh = meshes[mesh_kind]
    verify_input(mesh, protocol['mesh_sha256'][mesh_kind])
    for name, expected in protocol['input_sha256'].items():
        verify_input(inputs / name, expected)
    if args.check_only:
        for path in ROOT.rglob('*.py'):
            if 'runs' not in path.relative_to(ROOT).parts:
                compile(path.read_bytes(), str(path), 'exec')
        print(f'Wool {args.mode}: inputs and Python sources verified')
        return

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = args.output.resolve() if args.output else ROOT / 'runs' / f'{args.mode}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    lifetime = Lifetime(out / 'queue.lock')
    try:
        if args.mode == 'shear':
            settings = protocol['shear']
            case = settings['case'].copy()
            for stage in settings['stages']:
                folder = out / stage['name']
                prepare_stage(folder, ['shear'], mesh)
                shutil.copyfile(inputs / '61_Wool_warp_twist.npy', folder / 'data.npy')
                save(folder / 'case.json', dict(case, active_count=stage['active_count'],
                     training_frames=stage['frames'], epochs=args.epochs or settings['max_epochs']))
                run_stage(folder, stage['name'] + '.py', lifetime)
                result = load(folder / 'status.json')
                if result['status'] != 'complete':
                    raise RuntimeError(f"Shear stage did not complete: {folder}")
                case['initial'] = result['best_parameters']
            return

        epochs = args.epochs or protocol['epochs']
        initial = load(ROOT / 'joint' / 'initial.json')
        if args.mode == 'bending':
            parameters = {}
            for stage in ('warp3', 'warp5', 'weft5'):
                folder = out / stage
                prepare_stage(folder, ['isotropic'], mesh)
                direction = 'weft' if stage == 'weft5' else 'warp'
                shutil.copyfile(inputs / f'61_Wool_{direction}_compression.npy', folder / 'data.npy')
                shutil.copyfile(folder / 'source_snapshot.py', folder / 'template.py')
                run_stage(folder, stage + '.py', lifetime, epochs)
                result = load(folder / 'training' / 'status.json')
                if result['status'] != 'complete':
                    raise RuntimeError(f"Isotropic stage did not complete: {folder}")
                if stage != 'warp3':
                    parameters.update({f'{key}_{direction}': value for key, value in result['best_parameters'].items()})
            initial = dict(parameters=parameters, source='isotropic fits from this run')

        folder = out / 'orthotropic'
        prepare_stage(folder, ['joint'], mesh)
        for direction in ('warp', 'weft'):
            name = f'61_Wool_{direction}_compression.npy'
            shutil.copyfile(inputs / name, folder / name)
        save(folder / 'initial.json', initial)
        run_stage(folder, 'train_full_joint_copy.py', lifetime, epochs)
    finally:
        lifetime.close()


if __name__ == '__main__':
    main()
