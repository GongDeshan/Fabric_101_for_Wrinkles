import torch
import torch.nn
from torch.optim import Optimizer
import uipc
from typing import Callable, Tuple, Union
import uipc.backend as backend
import uipc.builtin as builtin
from uipc.diff_sim import (AdjointMethodFeature, 
                           EnableGradFeature,
                           LearningRateFeature,
                           UpdateParameterFeature)
from torch.optim.lr_scheduler import LRScheduler as BaseLRScheduler

class BufferUtils:
    @staticmethod
    def tensor2bufferview(tensor:Union[torch.Tensor]) -> backend.BufferView:
        return backend.BufferView(
            handle=tensor.contiguous().data_ptr(), 
            element_offset=0,
            element_count=tensor.numel(),
            element_size=tensor.element_size(),
            element_stride=tensor.element_size(),
            backend_name=builtin.none) # cpu only

class DiffSimParameter(torch.nn.Module):
    def __init__(self, scene:uipc.Scene, size:int, u2p:Callable[[torch.Tensor], torch.Tensor] = None):
        super(DiffSimParameter, self).__init__()
        self._scene:uipc.Scene = scene
        self._U = torch.nn.Parameter(torch.zeros(size, dtype=torch.float64), requires_grad=True)
        self._u2p = u2p
        
        self._world:uipc.World = None # world is not set yet
        self._update_parm_feature:UpdateParameterFeature = None # update parameter feature not set yet
        
        diff_sim = self._scene.diff_sim()
        parms = diff_sim.parameters()
        assert parms.size() == 0, f'when using DiffSimModule, diff_sim.parameters() must be empty, but we get parameters.size() = {parms.size()}'
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
        
    def __repr__ (self):
        U = self._U
        P = self.forward()
        return f'''DiffSimParameter:
    U: {U.__repr__()}
    P: {P.__repr__()}'''

class DiffSimModule(torch.nn.Module):
    class F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, module, P:torch.Tensor) -> torch.Tensor:
            ctx.module = module
            ctx.save_for_backward(P)
            self:DiffSimModule = module
            return self._X
    
        @staticmethod
        def backward(ctx, dLdX) -> Tuple[None, torch.Tensor]:
            self:DiffSimModule = ctx.module
            P:torch.Tensor = ctx.saved_tensors[0]
            dLdP = torch.empty(P.size(0), dtype=torch.float64)
            t2b = BufferUtils.tensor2bufferview
            self._adjoint_method.compute_dLdP(
                out_dLdP=t2b(dLdP),
                in_dLdX=t2b(dLdX))
            return None, dLdP

    class DofSelectInfo:
        def __init__(self, frame:int):
            self._frame:int = frame
        
        def frame(self):
            return self._frame

    def __init__(self, world:uipc.World, end_frame:int, parm:DiffSimParameter,
                 dof_select:Callable[[DofSelectInfo], Union[torch.Tensor, None]]):
        super(DiffSimModule, self).__init__()
        
        # 1. Basic setup
        self._parm = parm
        self._parm._world = world
        self._parm._update_parm_feature = world.features().find(UpdateParameterFeature)
        assert self._parm._update_parm_feature is not None,\
            f'''Cannot find feature {UpdateParameterFeature.FeatureName} in the world, world.features() = {world.features()}'''
        self.register_parameter('U', parm.U())
        self._world:uipc.World = world
        self._scene:uipc.Scene = parm._scene
        self._X:torch.Tensor = None # selected dofs
        
        assert end_frame >= 1, f'end_frame must be >= 1, got {end_frame}'
        self._end_frame:int = end_frame
        assert dof_select is not None, 'dof_select must be provided'
        self._dof_select = dof_select
        
        # 2. Dump the 0-th frame
        assert self._world.frame() == 0, f'''World frame must be 0 to start, but got {self._world.frame()}'''
        self._world.dump()
        
        # 3. Get Features
        features = self._world.features()
        self._adjoint_method:AdjointMethodFeature = features.find(AdjointMethodFeature)
        assert self._adjoint_method is not None,\
            f'''Cannot find feature {AdjointMethodFeature.FeatureName} in the world, world.features() = {features}'''
        self._enable_grad_feature:EnableGradFeature = features.find(EnableGradFeature)
        assert self._enable_grad_feature is not None,\
            f'''Cannot find feature {EnableGradFeature.FeatureName} in the world, world.features() = {features}'''
        
        # 4. Sync the parameter
        self._parm.sync()
        self.forward_func = DiffSimModule.F.apply
    
    def forward(self) -> torch.Tensor:
        t2b = BufferUtils.tensor2bufferview
        
        self._world.recover(0)
        P = self._parm.sync()
        
        # 1. Run the simulation
        total_sdi_count = 0
        if torch.is_grad_enabled():
            self._enable_grad_feature.with_grad()
        else:
            self._enable_grad_feature.no_grad()

        while self._world.frame() < self._end_frame:
            self._world.advance()
            F = self._world.frame()

            # 1) Select Dofs in Interest
            info = DiffSimModule.DofSelectInfo(F)
            sdi = self._dof_select(info)
            if sdi is None:
                sdi = torch.empty(0, dtype=torch.int32)
            else:
                assert sdi.dim() == 1, f'selected_indices must be 1-d tensor, got {sdi.dim()}'
                assert sdi.dtype == torch.int32, f'selected_indices must be of type int32, got {sdi.dtype}'
                total_sdi_count += sdi.shape[0]
            self._adjoint_method.select_dofs(F, t2b(sdi))

        # 2. Collect Dofs in Interest
        self._X = torch.empty(total_sdi_count, dtype=torch.float64, device=P.device, requires_grad=torch.is_grad_enabled())
        self._adjoint_method.receive_dofs(t2b(self._X))
        self._enable_grad_feature.no_grad() # close the grad anyway
        return self.forward_func(self, P)
    
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
            self._world.recover(0) # must recover to frame 0
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
        self._learning_rate_feature:LearningRateFeature = self._world.features().find(LearningRateFeature)
        assert self._learning_rate_feature is not None,\
            f'''Cannot find feature {LearningRateFeature.FeatureName} in the world, world.features() = {self._world.features()}'''
        
        self._old_P:torch.Tensor = None # old parameter
        self._old_U:torch.Tensor = None
        # record the old parameter
        with torch.no_grad():
            self._old_P:torch.Tensor = module._parm.forward()
            self._old_U:torch.Tensor = module._parm.U().detach().clone()
        
        super(DiffSimLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> torch.Tensor:
        return self.base_lrs # no change in learning rate
        