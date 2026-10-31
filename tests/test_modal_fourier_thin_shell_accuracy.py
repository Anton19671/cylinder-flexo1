"""Thin-shell Fourier modal spectral solver numerical accuracy checks."""
import numpy as np
import pytest
from cylinder_modal_fourier_fdm import (
    _quotient,_relative_eigen_quality,_refine_fourier_eigenpair,_certify_full_mode)


def test_eigenvector_rayleigh_value_overrides_bad_eigenvalue():
    k=np.diag([.05,2.e11]);m=np.diag([1.e-8,1.]);v=np.array([1.,0.])
    assert abs(_quotient(k,m,v)-5.000006e6)>5.
    value,vec,quality=_refine_fourier_eigenpair(k,m,5.000006e6,v)
    assert value==pytest.approx(5.e6,abs=1.e-8)
    assert quality<=5.e-9
    assert _relative_eigen_quality(k@vec,m@vec,value)<=5.e-9


def test_shift_invert_fixes_mixed_eigenvector():
    k=np.diag([1.,3.,15000.]);m=np.eye(3);v=np.array([1.,1.e-3,0.])
    val,vec,quality=_refine_fourier_eigenpair(k,m,1.0003,v)
    assert abs(val-1.)<1.e-9
    assert quality<5.e-9
    assert abs(vec[1])<1.e-8


def test_physical_action_disagreement_is_bounded():
    km=np.diag([2.,10.]);m0=np.eye(2);v=np.array([1.,0.])
    class Operator:
        def __init__(self,func): self.func=func
        def __matmul__(self,x): return self.func(x)
    k=Operator(lambda x:km@x+np.array([0.,2.e-6])*x[0])
    m=Operator(lambda x:m0@x)
    physical,projected,disagreement=_certify_full_mode(k,m,v,2.,km@v,m0@v)
    assert physical<1.e-6 and projected==0. and disagreement<1.e-6


def test_physical_operator_large_disagreement_rejected():
    km=np.diag([2.,10.]);mm=np.eye(2);v=np.array([1.,0.])
    class Operator:
        def __init__(self,matrix): self.matrix=matrix
        def __matmul__(self,x): return self.matrix@x
    with pytest.raises(ValueError,match='MODAL_FOURIER_EIGENPAIR_VERIFICATION_FAILED'):
        _certify_full_mode(Operator(km+np.array([[0.,0.],[1.,0.]])),
                           Operator(mm),v,2.,km@v,mm@v)


def test_corrupted_eigenvector_rejected():
    km=np.diag([2.,11.]);mm=np.eye(2)
    v=np.array([1.,1.])/np.sqrt(2.)
    with pytest.raises(ValueError,match='MODAL_FOURIER_BLOCK_EIGENPAIR_NOT_VERIFIED'):
        _refine_fourier_eigenpair(km,mm,6.5,v,max_iterations=0)
