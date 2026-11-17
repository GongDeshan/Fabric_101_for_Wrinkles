import torch

def shear_stress(q,elastic,k0,k1):
    return 2*(k0+k1*q.abs())*elastic+k1*q.sign()*elastic.square()
