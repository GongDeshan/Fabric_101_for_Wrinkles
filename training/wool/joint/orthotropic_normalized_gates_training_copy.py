"""Differentiable rest-direction gates paired with normalized CUDA bending."""
from typing import NamedTuple
import torch


class DirectionalGates(NamedTuple):
    w: torch.Tensor
    v: torch.Tensor
    normalization: torch.Tensor
    yield_warp: torch.Tensor
    yield_weft: torch.Tensor
    threshold_warp: torch.Tensor
    threshold_weft: torch.Tensor


def effective_directional_gates(cos_phi, sin_phi, y0_warp, y0_weft, t_warp, t_weft):
    """Keep raw material tensors live: both output directions depend on both inputs.

    These are INITIAL yield intercepts. Add q_d*H_d/K_d separately; hardening
    remains independent. Apply normalization to forces/energy, never to history
    updates or their transition Jacobians.
    """
    w, v = cos_phi.square(), sin_phi.square()
    y0 = w*y0_warp+v*y0_weft
    threshold = w*t_warp+v*t_weft
    return DirectionalGates(w,v,1./(w.square()+v.square()),
                            w*y0,v*y0,w*threshold,v*threshold)
