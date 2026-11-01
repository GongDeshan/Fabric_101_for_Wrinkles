"""Training-target controls embedded in the existing stretch scene panel."""
from pathlib import Path
import ast
import gc
import numpy as np
from polyscope import imgui
import polyscope as ps


def read_target(path, count):
    target = np.load(path, allow_pickle=False)
    if target.shape != (2, count) or not np.issubdtype(target.dtype, np.number):
        raise ValueError(f'Expected numeric (2, {count}) data: displacement (mm), force (N).')
    if not np.isfinite(target).all():
        raise ValueError('Target contains NaN or infinity.')
    x = target[0]
    if x[-1] <= 0 or not np.all(np.diff(x) > 0) or not np.allclose(
            x, np.linspace(0, x[-1], count), rtol=1e-5, atol=1e-7):
        raise ValueError('Displacements must be uniform, strictly increasing, and start at zero.')
    return target


class StretchDataControls:
    def __init__(self, env):
        self.env = env
        directory = Path(env['__file__']).resolve().parent
        self.files = sorted((directory / 'data').glob('*.npy'))
        current = Path(env['GT_PATH']).resolve()
        self.index = next((i for i, p in enumerate(self.files) if p.resolve() == current), len(self.files))
        self.path = str(current)
        self.message = ''
        self.density = float(env['MASS_DENSITY'])
        self.thickness = float(env['THICKNESS'])
        self.stiffness = float(env['K22'])
        self.learning_rate = float(env['PARAM_LR'])
        self.epochs = int(env['NUM_EPOCH'])

    def apply(self):
        e = self.env
        path = Path(self.path).expanduser().resolve(strict=True)
        read_target(path, e['EndFrame'] + 1)
        values = [self.density, self.thickness, self.stiffness, self.learning_rate]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('Density, thickness, initial stiffness and learning rate must be finite and positive.')
        if self.epochs < 1:
            raise ValueError('Training epochs must be positive.')
        direction = e['sample_vertial_direction']
        for direction in ('warp', 'weft'):
            if direction in path.stem.lower().split('_'):
                break
        else:
            direction = e['sample_vertial_direction']
        setup = dict(GT_PATH=str(path), sample_vertial_direction=direction,
            MASS_DENSITY=self.density, THICKNESS=self.thickness, K22=self.stiffness,
            PARAM_LR=self.learning_rate, NUM_EPOCH=self.epochs)
        e['_stretch_pending_setup'] = setup
        e['run'] = False
        ps.unshow()
        return True

    def draw(self):
        e = self.env
        imgui.Text('Training data')
        changed, self.index = imgui.Combo('Dataset', self.index,
            [p.name for p in self.files] + ['Custom path'])
        if changed and self.index < len(self.files):
            self.path = str(self.files[self.index])
        changed, self.path = imgui.InputText('Data file (.npy)', self.path, max_str_len=4096)
        if changed:
            self.index = len(self.files)
        _, self.density = imgui.InputFloat('Density (kg/m^3)', self.density, format='%.8g')
        _, self.thickness = imgui.InputFloat('Thickness (m)', self.thickness, format='%.8g')
        _, self.stiffness = imgui.InputFloat('Initial axial stiffness (Pa)', self.stiffness, format='%.9g')
        _, self.learning_rate = imgui.InputFloat('Adam learning rate (U)', self.learning_rate, format='%.8g')
        _, self.epochs = imgui.InputInt('Training epochs', self.epochs)
        if imgui.Button('Apply settings and reset scene'):
            try:
                return self.apply()
            except Exception as exc:
                self.message = 'Cannot load: ' + str(exc)
        imgui.Text('Active: ' + Path(e['GT_PATH']).name)
        imgui.Text(f"Density={e['MASS_DENSITY']:g} kg/m^3; thickness={e['THICKNESS']:g} m")
        imgui.Text(f"Active initial stiffness={e['K22']:g} Pa; LR={e['PARAM_LR']:g}; epochs={e['NUM_EPOCH']}")
        imgui.Text('Apply rebuilds the scene and resets all histories and Adam. Other constants stay unchanged.')
        if self.message:
            imgui.Text(self.message)
        return False


def configured_scene(source, setup):
    tree = ast.parse(source)
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in setup:
                    node.value = ast.Constant(setup[target.id])
                    found.add(target.id)
    if found != set(setup):
        raise ValueError(f'Missing scene settings: {set(setup) - found}')
    return ast.fix_missing_locations(tree)


def run_scene_window(model_path):
    """Rebuild the complete backend when physical constants change, reusing the GUI."""
    model = Path(model_path).resolve()
    source = model.read_text(encoding='utf-8-sig')
    setup = {}
    ps.init()
    while True:
        env = dict(__name__='__main__', __file__=str(model),
            _stretch_embedded_scene=True, _stretch_polyscope_initialized=True)
        next_setup = None
        try:
            exec(compile(configured_scene(source, setup), str(model), 'exec'), env)
            next_setup = env.get('_stretch_pending_setup')
        finally:
            from stretch_logging_theme import set_logging_theme
            set_logging_theme(True)
            ps.set_user_callback(lambda: None)
            ps.remove_all_structures()
            env.clear()
            gc.collect()
        if next_setup is None:
            break
        setup = next_setup
        pass  
