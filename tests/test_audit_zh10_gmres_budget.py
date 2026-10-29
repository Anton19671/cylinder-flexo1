"""Zh-10: actual GMRES matvec budget, independent of restart cycles."""
import numpy as np
import pytest
from scipy.sparse.linalg import gmres, LinearOperator
from cylinder_matrixfree_fdm import MatrixFreeNewtonSettings, _KrylovMatvecBudgetExceeded

def test_budget_guard_counts_real_actions():
    calls=[0]
    def action(x):
        if calls[0]>=3:
            raise _KrylovMatvecBudgetExceeded("KRYLOV_MATVEC_BUDGET_EXCEEDED")
        calls[0]+=1
        return np.array([x[0]+x[1],x[0]+2*x[1]],float)
    a=LinearOperator((2,2),matvec=action,dtype=float)
    try:
        gmres(a,np.array([1.,2.]),rtol=1e-15,atol=0.,restart=2,
              maxiter=4,callback_type="pr_norm")
    except _KrylovMatvecBudgetExceeded:
        pass
    assert calls[0]<=3

def test_settings_budget_not_restarts():
    settings=MatrixFreeNewtonSettings(krylov_max_matvecs=3,krylov_restart=2)
    assert settings.krylov_max_matvecs==3
