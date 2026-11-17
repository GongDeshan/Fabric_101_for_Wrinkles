"""Energy-consistent Python readout for the NEW local-yarn-angle constitution."""
import torch

PARAMETER_ORDER=('k11','k22','k12','k33_0','k33_1','friction_k33',
                 'thres_uv','yield_ori_uv','hardening_k33')

def deformation_gradient(x,rest_uv):
    ds=torch.stack((x[...,1,:]-x[...,0,:],x[...,2,:]-x[...,0,:]),dim=-1)
    dm=torch.stack((rest_uv[...,1,:]-rest_uv[...,0,:],rest_uv[...,2,:]-rest_uv[...,0,:]),dim=-1)
    return ds@torch.linalg.inv(dm)

def yarn_angle(x,rest_uv):
    F=deformation_gradient(x,rest_uv)
    u,v=F[...,0],F[...,1]
    return torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(u,v,dim=-1),dim=-1),
                       (u*v).sum(dim=-1))

def effective_coefficients(G,parameters,enabled=True):
    k11,k22,k12,k0,k1=parameters[...,0],parameters[...,1],parameters[...,2],parameters[...,3],parameters[...,4]
    gu,gv,q=G[...,0],G[...,1],G[...,2]
    k=k0+k1*q.abs()
    if not enabled:return k11,k22,k12,k
    denominator=(1+2*gu)*(1+2*gv)
    t=torch.where(denominator>1e-24,4*q.square()/denominator.clamp_min(1e-24),torch.zeros_like(q))
    t=t.clamp(0,1-1e-12);u=1-t
    return k11+2*k12*t+k22*t.square()+4*k*t,k22*u.square(),k12*u+k22*t*u,k22*t*u+k*u

def strain(x,rest_uv):
    F=deformation_gradient(x,rest_uv)
    C=F.transpose(-1,-2)@F
    return torch.stack((.5*(C[...,0,0]-1),.5*(C[...,1,1]-1),.5*C[...,0,1]),dim=-1)

def energy(x,rest_uv,parameters,plastic,anchor_uv,enabled=True):
    G=strain(x,rest_uv)
    k11,k22,k12,k=effective_coefficients(G,parameters,enabled)
    e=G-plastic
    return .5*k11*e[...,0].square()+k12*e[...,0]*e[...,1]+.5*k22*e[...,1].square() \
        +k*e[...,2].square()+parameters[...,5]*(G[...,2]-anchor_uv).square()

def return_mapping(G,parameters,previous_plastic,previous_anchor,previous_hardening,enabled=True):
    """Same prediction-stage local update as CUDA, suitable for a history replay."""
    k=effective_coefficients(G,parameters,enabled)[3].clamp_min(1e-12)
    kh=parameters[...,8]
    q=G[...,2];elastic=q-previous_plastic
    yield_value=parameters[...,7]+previous_hardening*kh/k
    increment=torch.relu(elastic.abs()-yield_value)*k/(k+kh)
    plastic=previous_plastic+elastic.sign()*increment
    hardening=previous_hardening+increment
    distance=q-previous_anchor
    anchor=previous_anchor+distance.sign()*torch.relu(distance.abs()-parameters[...,6])
    return plastic,anchor,hardening,parameters[...,7]+hardening*kh/k

def nodal_force(x,rest_uv,parameters,plastic,anchor_uv,rest_area,thickness,enabled=True,create_graph=True):
    """Physical nodal forces; dt^2 belongs only to the backend objective."""
    if not x.requires_grad:x=x.detach().requires_grad_(True)
    E=(energy(x,rest_uv,parameters,plastic,anchor_uv,enabled)*rest_area*thickness).sum()
    return -torch.autograd.grad(E,x,create_graph=create_graph,retain_graph=True)[0]
