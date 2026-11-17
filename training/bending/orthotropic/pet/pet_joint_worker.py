"""Independent cold-process normalized orthotropic worker: full material model."""
import ast
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'LUFHES_elastic_stvk_lr_compress_orthotropic_normalized_training_copy.py'
ACTIVE = ('kb_warp', 'kb_weft', 'friction_warp', 'friction_weft', 'hardening_warp', 'yield_warp', 'thres_warp', 'thres_weft')
def save(path, value):
    tmp=path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8'); tmp.replace(path)
def prepare_source(direction, out):
    source = MODEL.read_text(encoding="utf-8-sig")
    source = source.rsplit("\nif _cli_args.audit_position_frames is not None:\n", 1)[0]
    changes = {
        '_log_dir = os.path.join(os.path.dirname(__file__), "logs")':
            "_log_dir = " + repr(str(out / "model_logs")),
        "P = self._lower + (self._upper - self._lower) * torch.sigmoid(self.U())":
            "P = self.U().square()",
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    old_bounds = "(10.0, 10.0, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-8, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4)"
    assert source.count(old_bounds) == 1
    source = source.replace(old_bounds, old_bounds.replace("10.0, 10.0", "1.0e-8, 1.0e-8"))
    assert source.count("(5000.0, 5000.0,") == 1
    source = source.replace("(5000.0, 5000.0,", "(1.0e12, 1.0e12,")
    from train_remaining_bending_pipeline import assign
    source=assign(source, {"MESH_PATH": repr(str(ROOT.parent / "mesh.obj"))})
    data_lines = [line for line in source.splitlines() if "with open(f" in line and "compression.npy" in line]
    assert len(data_lines) == 1
    data_dir = ROOT if direction == "warp" else ROOT.parent.parent / "isotropic" / "data"
    data_path = data_dir / f"62_pet_{direction}_compression.npy"
    source = source.replace(data_lines[0], f"with open({str(data_path)!r}, 'rb') as f:")
    point = 'gt_comp_force_all = torch.from_numpy'
    assert source.count(point) == 1
    source = source.replace(point, '_training_frames = np.r_[30:71, 120:151].astype(np.int32)\n' + point)
    # P=U^2 has no sigmoid clipping; permit positive transfers outside old bounds.
    source=assign(source, {'_BENDING_PARAMETER_LOWER': repr(tuple([-1.0]*10)),
                           '_BENDING_PARAMETER_UPPER': repr(tuple([float(1e30)]*10))})
    assert 'strain_uus = G[0] - strain_uus_init' in source
    assert 'strain_vvs = G[3] - strain_vvs_init' in source
    assert 'strain_uvs = G[1]\n' in source
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == "MyDiffSimParameter")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "set_physical")
    lines = source.splitlines(True)
    lines[method.lineno-1:method.end_lineno] = [
        "    def set_physical(self, values):\n",
        "        values = torch.as_tensor(values, dtype=torch.float64)\n",
        "        if not torch.isfinite(values).all() or not (values >= 0).all():\n",
        "            raise ValueError('Invalid physical parameters')\n",
        "        self.U().copy_(values.sqrt())\n"]
    return "".join(lines)


def worker(request, response):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    req = json.loads(request.read_text(encoding="utf-8"))
    out = request.parent
    source = prepare_source(req["direction"], out)
    executed = out / "executed_model.py"
    executed.write_text(source, encoding="utf-8")
    sys.argv = [str(MODEL), "--headless-train", "--fabric", "62_pet",
                "--stage", req["direction"], "--mechanism-phase", "refine",
                "--training-contact", "--bending-history-autograd", "--training-end-frame", "150",
                "--loss-frame-range", "30", "151", "--solver-tolerances", ".005", ".0001",
                *[item for name in ACTIVE for item in ("--active-parameter",name)], "--epochs", "1"]
    for name, value in req["parameters"].items():
        sys.argv += ["--initial-parameter", f"{name}={value:.17g}"]
    ns = dict(__name__="joint_square_worker", __file__=str(executed))
    exec(compile(source, str(executed), "exec"), ns)
    assert ns["config"]["contact"]["enable"]
    assert ns["kb_33"]==0.0 and ns["kb_12"]==0.0
    assert ns["_cli_args"].bending_history_autograd
    assert "bending_normalization" in ns["net_force"].__code__.co_varnames
    assert ns["MESH_PATH"] == str(ROOT.parent / "mesh.obj")
    assert ns["NUM_VERTICES"]==1820 and ns["NUM_FACES"]==3456
    assert ns["tested_fabric_type"]=='62_pet'
    assert ns["_active_parameter_names"] == ACTIVE
    signed_u = torch.tensor(req["signed_U"], dtype=torch.float64)
    assert signed_u.shape == (10,) and torch.isfinite(signed_u).all()
    np.testing.assert_allclose(signed_u.square().numpy(),
        [req["parameters"][n] for n in ('kb_warp', 'kb_weft', 'friction_warp', 'friction_weft', 'hardening_warp', 'hardening_weft', 'yield_warp', 'yield_weft', 'thres_warp', 'thres_weft')], rtol=1e-12, atol=1e-12)
    with torch.no_grad():
        ns["parm"].U()[:10].copy_(signed_u)
        ns["parm"].sync()
    assert torch.equal(ns["parm"].U()[:10].detach().cpu(), signed_u)
    loss = ns["loss_fn"](training=req["gradient"], verbose=False)
    assert torch.equal(ns["parm"].U()[:10].detach().cpu(), signed_u), "Rollout changed signed U"
    actual = ns["parm"].forward().detach().numpy()
    np.testing.assert_allclose(actual, [req["parameters"][n] for n in ns["_BENDING_PARAMETER_NAMES"]],
                               rtol=1e-12, atol=1e-12)
    module = ns['diff_module']
    plastic = {}
    for direction in ('warp', 'weft'):
        values = np.asarray(getattr(module, 'get_plastic_thetas_' + direction)())
        rest = getattr(module, '_rest_plastic_thetas_' + direction)
        hardening = np.asarray(getattr(module, 'get_hardening_plastic_thetas_' + direction)())
        plastic[direction] = {'max_change': float(np.max(np.abs(values - rest))),
                              'hardening_max': float(np.max(np.abs(hardening)))}
        assert np.isfinite(values).all() and np.isfinite(hardening).all() and np.min(hardening) >= -1e-10, plastic
    assert np.array_equal(ns['_training_frames'], np.r_[30:71, 120:151])
    result = dict(loss=float(loss.detach()), parameters=req["parameters"], signed_U=signed_u.tolist(),
                  prediction=ns["last_prediction"].tolist(),
                  gt=ns["gt_comp_force"].detach().numpy().tolist())
    result["plastic_check"] = ({'warp': plastic['weft'], 'weft': plastic['warp']}
                               if req['direction'] == 'warp' else plastic)
    result["frames"] = ns["_training_frames"].tolist()
    if req["gradient"]:
        result["gradient_U"] = ns["parm"].U().grad.detach().numpy()[:10].tolist()
    save(response, result)

if __name__ == '__main__':
    assert sys.stdin.readline().strip() == 'START'
    worker(Path(sys.argv[1]), Path(sys.argv[2]))
