"""Differentiable replay of CUDA's bending-history return mapping.

Replays the history readout only, not the mechanical solve. X keeps its native
adjoint, so dL/dhistory is routed through earlier X as well as directly to P.
Never uses finite differences to supply training gradients.
"""
import numpy as np
import torch


def replay_bending_history(context, X, model_P, recorded, *, check=True):
    n = context['NUM_VERTICES']
    trajectory = X.reshape(-1, n, 3)
    initial = torch.as_tensor(context['_bending_history_initial_positions'], dtype=X.dtype,
                              device=X.device).reshape(n, 3)
    valid_np = np.asarray(context['egde_rest_lengths']).reshape(-1) != 0
    ids = torch.as_tensor(np.flatnonzero(valid_np), device=X.device)
    stencil = torch.as_tensor(np.asarray(context['bending_edge_stencils']).reshape(-1, 4)[valid_np],
                              dtype=torch.long, device=X.device)
    edge_bias = {tuple(sorted(map(int, edge))): float(angle)
                 for edge, angle in zip(np.asarray(context['edge_vert_topos']).reshape(-1, 2),
                                        np.asarray(context['bias_angles']).reshape(-1))}
    bias = np.asarray([edge_bias[tuple(sorted(map(int, edge)))]
                       for edge in stencil.cpu().numpy()[:, 1:3]])
    phi = torch.as_tensor(bias, dtype=X.dtype, device=X.device) + 1.570796
    weights = (torch.cos(phi).square(), torch.sin(phi).square())
    module = context['diff_module']
    outputs = {key: [] for key in ('plastic_warp', 'plastic_weft', 'anchor_warp',
               'anchor_weft', 'yield_warp', 'yield_weft', 'hardening_warp', 'hardening_weft')}
    states = []
    for direction, k_idx in (('warp', 0), ('weft', 1)):
        p = torch.as_tensor(getattr(module, '_rest_plastic_thetas_'+direction),
                            dtype=X.dtype, device=X.device).reshape(-1)[ids]
        a = torch.as_tensor(getattr(module, '_rest_anchor_thetas_'+direction),
                            dtype=X.dtype, device=X.device).reshape(-1)[ids]
        q = torch.zeros_like(p)
        y = weights[k_idx] * (weights[0]*model_P[6] + weights[1]*model_P[7]) + q*model_P[4+k_idx]/model_P[k_idx]
        states.append((p, a, q, y))
    for frame in range(len(trajectory)):
        previous = initial if frame == 0 else trajectory[frame-1]
        p0, u, v, p3 = previous[stencil].unbind(dim=1)
        n1 = torch.linalg.cross(u-p0, v-p0)
        n2 = torch.linalg.cross(v-p3, u-p3)
        edge = u-v
        length = torch.linalg.vector_norm(edge, dim=1)
        if bool((length == 0).any()) or bool((torch.linalg.vector_norm(n1, dim=1) == 0).any()) or bool((torch.linalg.vector_norm(n2, dim=1) == 0).any()):
            raise RuntimeError('Degenerate hinge in history replay')
        sine = (torch.linalg.cross(n2, n1)*edge).sum(dim=1)/length
        theta = torch.atan2(sine, (n1*n2).sum(dim=1))
        for d, direction in enumerate(('warp', 'weft')):
            p, a, q, y = states[d]
            k, h, y0, threshold = (model_P[d], model_P[4+d], model_P[6+d], model_P[8+d])
            elastic = theta*weights[d]-p
            y0_eff = weights[d] * (weights[0]*model_P[6] + weights[1]*model_P[7])
            threshold = weights[d] * (weights[0]*model_P[8] + weights[1]*model_P[9])
            trial_yield = y0_eff+q*h/k
            active = elastic.abs() > trial_yield
            gamma = torch.where(active, (elastic.abs()-trial_yield)*k/(k+h),
                                torch.zeros_like(elastic))
            p = p+torch.sign(elastic)*gamma
            q = q+gamma
            y = y0_eff+q*h/k
            relative = theta*weights[d]-a
            slide = relative.abs() > threshold
            a = a+torch.where(slide, torch.sign(relative)*(relative.abs()-threshold),
                              torch.zeros_like(relative))
            states[d] = (p, a, q, y)
            for key, val in (('plastic', p), ('anchor', a), ('yield', y), ('hardening', q)):
                name = key+'_'+direction
                base = recorded[name][frame].detach().reshape(-1)
                outputs[name].append(base.index_copy(0, ids, val).reshape_as(recorded[name][frame]))
    result = {key: torch.stack(values) for key, values in outputs.items()}
    errors = {key: float((value.detach()-recorded[key]).abs().max())
              for key, value in result.items()}
    if check and max(errors.values()) > 1e-7:
        raise RuntimeError(f'History replay differs from CUDA; refusing gradient: {errors}')
    return result, errors
