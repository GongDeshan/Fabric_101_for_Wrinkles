import sympy as sp
from SymEigen import *

def compute_dTdXn(T:EigenMatrix, Xn:EigenMatrix):
    N = Xn.shape[0]
    assert Xn.shape[1] == 1
    assert T.shape == (2, N)
    dTdXn :sp.NDimArray = derive_by_array(T, Xn)
    assert dTdXn.shape == (N, 1, 2, N)
    dTdXn = dTdXn.reshape(N, 2, N)
    return dTdXn


def compute_AB(dTdXn:sp.NDimArray, dXp:EigenMatrix, dfdtau:EigenMatrix):
    N = dXp.shape[0]
    assert dfdtau.shape == (2, 1)
    assert dTdXn.shape == (N, 1, 2, N) or dTdXn.shape == (N, 2, N)
    if(dTdXn.shape == (N, 1, 2, N)):
        dTdXn = dTdXn.reshape(N, 2, N)
    tp = sp.tensorproduct(dTdXn, dXp)
    assert tp.shape == (N, 2, N, N, 1)
    tp = tp.reshape(N, 2, N, N)
    A = sp.tensorcontraction(tp, (2,3))
    A = A.tomatrix().T
    assert A.shape == (2, N)
    
    tp = sp.tensorproduct(dTdXn, dfdtau)
    assert tp.shape == (N, 2, N, 2, 1)
    tp = tp.reshape(N, 2, N, 2)
    B = sp.tensorcontraction(tp, (1,3))
    B = B.tomatrix().T
    assert B.shape == (N, N)
    
    return A, B