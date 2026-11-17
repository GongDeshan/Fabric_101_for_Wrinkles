"""Launch the public training workflows with one fabric and stage selector."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
FABRICS = {
    'cotton': '59_Cotton',
    'linen': '60_Linen',
    'wool': '61_Wool',
    'pet': '62_pet',
    'silk': '63_Silk',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fabric', choices=FABRICS, required=True)
    parser.add_argument('--stage', choices=('stretch', 'shear', 'bending', 'joint'), required=True)
    parser.add_argument('--direction', choices=('warp', 'weft'), help='Stretch or shear direction; defaults to warp')
    parser.add_argument('--isotropic-run', type=Path, help='Saved isotropic run for a Cotton or Linen joint fit')
    parser.add_argument('--check-only', action='store_true', help='Check the selected files and show the command')
    args = parser.parse_args()

    fabric = args.fabric
    stage = args.stage
    fabric_id = FABRICS[fabric]
    direction = args.direction or 'warp'
    bending = ROOT / 'bending'
    paths = []
    if args.direction and stage not in ('stretch', 'shear'):
        parser.error('--direction applies to stretch and shear')
    if args.isotropic_run and not (stage == 'joint' and fabric in ('cotton', 'linen')):
        parser.error('--isotropic-run applies to Cotton and Linen joint training')

    if stage == 'stretch':
        script = ROOT / 'stretch' / 'stretch_model.py'
        options = ['--fabric', fabric_id, '--direction', direction]
        paths.append(script.parent / 'data' / f'{fabric_id}_{direction}_friction.npy')
    elif fabric == 'wool' and (stage != 'shear' or direction == 'warp'):
        script = ROOT / 'wool' / 'train.py'
        options = [stage]
    elif stage == 'shear':
        script = ROOT / 'shear' / 'train.py'
        options = ['--fabric', fabric_id, '--direction', direction]
        paths.append(script.parent / 'data' / f'{fabric_id}_{direction}_twist.npy')
    elif fabric in ('pet', 'silk'):
        script = bending / 'orthotropic' / fabric / 'train.py'
        options = []
    elif stage == 'bending':
        script = bending / 'isotropic' / 'train.py'
        options = ['--fabric', fabric_id]
        paths.extend(script.parent / 'data' / f'{fabric_id}_{d}_compression.npy' for d in ('warp', 'weft'))
    else:
        script = bending / 'orthotropic' / 'train.py'
        options = ['--isotropic-run', str(args.isotropic_run.resolve())] if args.isotropic_run else []

    for path in [script, *paths]:
        if not path.is_file():
            parser.error(f'File not found: {path}')
    command = [sys.executable, str(script), *options]
    if args.check_only:
        print(subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command))
        return 0
    return subprocess.call(command, cwd=script.parent)


if __name__ == '__main__':
    raise SystemExit(main())

