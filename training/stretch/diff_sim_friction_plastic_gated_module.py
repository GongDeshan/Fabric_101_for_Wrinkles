import torch
import os
import time
from torch.optim import Optimizer

import numpy as np

from uipc.core import SceneIO
import uipc
from uipc import view
import uipc.backend as backend
from typing import Callable, Tuple, Union
import uipc.builtin as builtin
from uipc.diff_sim import (AdjointMethodFeature,
                           EnableGradFeature,
                           LearningRateFeature,
                           UpdateParameterFeature)
from torch.optim.lr_scheduler import LRScheduler as BaseLRScheduler


class BufferUtils:
    @staticmethod
    def tensor2bufferview(tensor: Union[torch.Tensor]) -> backend.BufferView:
        return backend.BufferView(
            handle=tensor.contiguous().data_ptr(),
            element_offset=0,
            element_count=tensor.numel(),
            element_size=tensor.element_size(),
            element_stride=tensor.element_size(),
            backend_name=builtin.none)  

class DiffSimParameter(torch.nn.Module):
    def __init__(self, scene: uipc.Scene, size: int, u2p: Callable[[torch.Tensor], torch.Tensor] = None):
        super(DiffSimParameter, self).__init__()
        self._scene: uipc.Scene = scene
        self._U = torch.nn.Parameter(torch.zeros(size, dtype=torch.float64), requires_grad=True)
        self._u2p = u2p

        self._world: uipc.World = None  
        self._update_parm_feature: UpdateParameterFeature = None

        diff_sim = self._scene.diff_sim()
        parms = diff_sim.parameters()
        assert parms.size(
        ) == 0, f'when using DiffSimModule, diff_sim.parameters() must be empty, but we get parameters.size() = {parms.size()}'
        P = self.forward()
        parms.resize(P.size(0))

    def U(self):
        return self._U

    def forward(self) -> torch.Tensor:
        '''
        Subclass should implement this method to compute the simulator parameters
        '''
        U = self.U()
        if self._u2p is not None:
            return self._u2p(U)
        return U

    def sync(self):
        t2b = BufferUtils.tensor2bufferview
        P = self.forward()
        if self._update_parm_feature is None:
            parms = self._scene.diff_sim().parameters()
            assert parms.size() == P.size(0), f'sim_parms.size must be equal to P.size, got {parms.size()}'
            sim_parms = uipc.view(parms)
            sim_parms[:] = P.detach().cpu().numpy()
            self._scene.diff_sim().parameters().broadcast()
        else:
            self._update_parm_feature.update(t2b(P))
        return P

    def __repr__(self):
        U = self._U
        P = self.forward()
        return f'''DiffSimParameter:
    U: {U.__repr__()}
    P: {P.__repr__()}'''

class DiffSimModule(torch.nn.Module):

    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, module, P: torch.Tensor) -> torch.Tensor:
            ctx.module = module
            ctx.save_for_backward(P)
            self: DiffSimModule = module
            return self._X

        @staticmethod
        def backward(ctx, dLdX) -> Tuple[None, torch.Tensor]:
            self: DiffSimModule = ctx.module
            P: torch.Tensor = ctx.saved_tensors[0]
            dLdP = torch.empty(P.size(0), dtype=torch.float64)
            t2b = BufferUtils.tensor2bufferview
            self._adjoint_method.compute_dLdP(
                out_dLdP=t2b(dLdP),
                in_dLdX=t2b(dLdX))

            pass  
            self._last_dLdP = dLdP.detach().clone()
            return None, dLdP
        
    class DofSelectInfo:
        def __init__(self, frame:int):
            self._frame:int = frame
        
        def frame(self):
            return self._frame

    def __init__(self, world, geo_slot, end_frame, parm, dof_select):
        
        super(DiffSimModule, self).__init__()

        self._parm = parm
        self._parm._world = world
        self._parm._update_parm_feature = world.features().find(UpdateParameterFeature)
        assert self._parm._update_parm_feature is not None, \
            f'''Cannot find feature {UpdateParameterFeature.FeatureName} in the world, world.features() = {world.features()}'''
        self.register_parameter('U', parm.U())  

        self._world = world
        self._scene: uipc.Scene = parm._scene
        self._geo_slot = geo_slot
        self._X: torch.Tensor = None  

        self._sio = SceneIO(self._scene)

        assert end_frame >= 1, f'end_frame must be >= 1, got {end_frame}'
        self._end_frame: int = end_frame
        assert dof_select is not None, 'dof_select must be provided'
        self._dof_select = dof_select
        self._rest_plastic_thetas = self._geo_slot.geometry().edges().find("plastic_theta").view().copy()
        self._rest_anchor_thetas = self._geo_slot.geometry().edges().find("anchor_theta").view().copy()
        self._rest_yield_thetas = self._geo_slot.geometry().edges().find("yield_theta").view().copy()
        self._rest_anchor_uvs = self._geo_slot.geometry().triangles().find("anchor_strain_uv").view().copy()
        self._rest_yield_uvs = self._geo_slot.geometry().triangles().find("yield_uv").view().copy()
        self._rest_plastic_uvs = self._geo_slot.geometry().triangles().find("plastic_strain_uv").view().copy()

        self._anchor_thetas = []  
        self._plastic_thetas = []   
        self._yield_thetas = []  
        self._hardening_plastic_thetas = []   

        self._anchor_strain_uvs = []

        self._plastic_strain_uvs = []

        self._yield_strain_uvs = []

        self._hardening_strain_uvs = []

        assert self._world.frame() == 0, f'''World frame must be 0 to start, but got {self._world.frame()}'''
        if not self._world.dump():
            raise RuntimeError('Frame-0 dump failed; cannot safely recover training/FD rollouts')

        features = self._world.features()
        self._adjoint_method: AdjointMethodFeature = features.find(
            AdjointMethodFeature)
        assert self._adjoint_method is not None, \
            f'''Cannot find feature {AdjointMethodFeature.FeatureName} in the world, world.features() = {features}'''
        self._enable_grad_feature: EnableGradFeature = features.find(
            EnableGradFeature)
        assert self._enable_grad_feature is not None, \
            f'''Cannot find feature {EnableGradFeature.FeatureName} in the world, world.features() = {features}'''

        self._parm.sync()
        self.forward_func = DiffSimModule.F.apply

    def forward(self) -> torch.Tensor:

        t2b = BufferUtils.tensor2bufferview

        if not self._world.recover(0):
            raise RuntimeError('Frame-0 recovery failed; refusing a rollout from an unknown initial state')

        view(self._geo_slot.geometry().edges().find("reset_anchor_plastic"))[:] = 1
        view(self._geo_slot.geometry().triangles().find("reset_anchor_plastic"))[:] = 1

        view(self._geo_slot.geometry().edges().find("anchor_theta"))[:] = self._rest_anchor_thetas   
        view(self._geo_slot.geometry().edges().find("plastic_theta"))[:] = self._rest_plastic_thetas    
        view(self._geo_slot.geometry().edges().find("yield_theta"))[:] = self._rest_yield_thetas    

        view(self._geo_slot.geometry().triangles().find("yield_uv"))[:] = self._rest_yield_uvs
        view(self._geo_slot.geometry().triangles().find("anchor_strain_uv"))[:] = self._rest_anchor_uvs
        view(self._geo_slot.geometry().triangles().find("plastic_strain_uv"))[:] = self._rest_plastic_uvs
        
        self._world.retrieve()

        P = self._parm.sync()

        self._plastic_thetas.clear()
        self._anchor_thetas.clear()
        self._yield_thetas.clear()
        self._hardening_plastic_thetas.clear()

        self._anchor_strain_uvs.clear()
        self._plastic_strain_uvs.clear()
        self._yield_strain_uvs.clear()
        self._hardening_strain_uvs.clear()

        total_sdi_count = 0
        if torch.is_grad_enabled():
            self._enable_grad_feature.with_grad()
        else:
            self._enable_grad_feature.no_grad()

        max_abs_strain_uv = 0.0

        while self._world.frame() < self._end_frame:

            trace_frames = os.environ.get("UIPC_FRAME_TRACE") == "1"
            next_frame = self._world.frame() + 1
            if trace_frames:
                advance_started = time.perf_counter()
                pass  
            self._world.advance()
            if trace_frames:
                pass  

            F = self._world.frame()

            if F == 1:
                view(self._geo_slot.geometry().edges().find("reset_anchor_plastic"))[:] = 0
                view(self._geo_slot.geometry().triangles().find("reset_anchor_plastic"))[:] = 0

                plastic_uv_now = self._geo_slot.geometry().triangles().find("plastic_strain_uv").view()
                hardening_uv_now = self._geo_slot.geometry().triangles().find("hardening_plastic_uv").view()
                yield_uv_now = self._geo_slot.geometry().triangles().find("yield_uv").view()
                pass  

            strain_uv_now = self._geo_slot.geometry().triangles().find("strain_uv").view()
            max_abs_strain_uv = max(max_abs_strain_uv, float(np.max(np.abs(strain_uv_now))))

            self._plastic_thetas.append(self._geo_slot.geometry().edges().find("plastic_theta").view().copy())
            self._anchor_thetas.append(self._geo_slot.geometry().edges().find("anchor_theta").view().copy())
            self._yield_thetas.append(self._geo_slot.geometry().edges().find("yield_theta").view().copy())
            self._hardening_plastic_thetas.append(self._geo_slot.geometry().edges().find("hardening_plastic_theta").view().copy())

            self._anchor_strain_uvs.append(self._geo_slot.geometry().triangles().find("anchor_strain_uv").view().copy())
            self._plastic_strain_uvs.append(self._geo_slot.geometry().triangles().find("plastic_strain_uv").view().copy())
            self._yield_strain_uvs.append(self._geo_slot.geometry().triangles().find("yield_uv").view().copy())
            self._hardening_strain_uvs.append(self._geo_slot.geometry().triangles().find("hardening_plastic_uv").view().copy())

            info = DiffSimModule.DofSelectInfo(F)
            sdi = self._dof_select(info)
            if sdi is None:
                sdi = torch.empty(0, dtype=torch.int32)
            else:
                assert sdi.dim(
                ) == 1, f'selected_indices must be 1-d tensor, got {sdi.dim()}'
                assert sdi.dtype == torch.int32, f'selected_indices must be of type int32, got {sdi.dtype}'
                total_sdi_count += sdi.shape[0]
            self._adjoint_method.select_dofs(F, t2b(sdi))
            

        pass  

        self._X = torch.empty(total_sdi_count, dtype=torch.float64, device=P.device, requires_grad=torch.is_grad_enabled())
        self._adjoint_method.receive_dofs(t2b(self._X))
        self._enable_grad_feature.no_grad()  
        return self.forward_func(self, P)
    
    def get_anchor_strains(self):
        return self._anchor_strain_uvs
    
    def get_plastic_strains(self):
        return self._plastic_strain_uvs
    
    def get_yield_strains(self):
        return self._yield_strain_uvs
    
    def get_hardening_strains(self):
        return self._hardening_strain_uvs

    def get_plastic_thetas(self):
        return self._plastic_thetas
    
    def get_anchor_thetas(self):
        return self._anchor_thetas
    
    def get_yield_thetas(self):
        return self._yield_thetas
    
    def get_hardening_plastic_thetas(self):
        return self._hardening_plastic_thetas
  
    def __call__(self) -> torch.Tensor:
        return self.forward()

class DiffSimLR(BaseLRScheduler):
    def step_post_hook(self, optimizer:Optimizer, *args, **kwargs) -> None:
        """
        This method is called after each optimization step.
        It is used to perform any post-processing after the optimization step.
        """
        with torch.no_grad():
            new_P = self._module._parm.forward()
            new_U = self._module._parm.U().detach().clone()
            v2b = BufferUtils.tensor2bufferview
            self._world.recover(0) 
            alpha = self._learning_rate_feature.learning_rate(v2b(new_P))
            new_P = (1-alpha) * self._old_P + alpha * new_P
            new_U = (1-alpha) * self._old_U + alpha * new_U
            self._old_P.copy_(new_P)
            self._old_U.copy_(new_U)
            self._module._parm.U().data.copy_(new_U)
        
    def __init__(self, module:DiffSimModule, optimizer:Optimizer, last_epoch:int = -1):
        self._optimizer: Optimizer = optimizer
        self._optimizer.register_step_post_hook(lambda optimizer, *args, **kwargs: self.step_post_hook(optimizer, *args, **kwargs))
        
        self._module:DiffSimModule = module
        self._world:uipc.World = module._world
        self._geo_slot = module._geo_slot
        self._learning_rate_feature:LearningRateFeature = self._world.features().find(LearningRateFeature)
        assert self._learning_rate_feature is not None,\
            f'''Cannot find feature {LearningRateFeature.FeatureName} in the world, world.features() = {self._world.features()}'''
        
        self._old_P:torch.Tensor = None 
        self._old_U:torch.Tensor = None
        with torch.no_grad():
            self._old_P:torch.Tensor = module._parm.forward()
            self._old_U:torch.Tensor = module._parm.U().detach().clone()
        
        super(DiffSimLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> torch.Tensor:
        return self.base_lrs 
