"""Structural test: electric-only X/Y closure freezes reconstructed mechanics.

No physical shell calculation; a synthetic electric functional supplies
the local first variation. This guards the endpoint potential-only solve
used by independent MVI Gauss2 against changes to mechanical fields.
"""
from types import SimpleNamespace
import numpy as np
import pytest

from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,CoupledMVIFactors,NativeCoupledMVISurface,
    CoupledMVIWork,
)
from independent_mvi.coupled_halfstep import (
    CoupledMVIEndProgram,solve_coupled_mvi_halfstep,
)
from independent_mvi.coupled_static import solve_coupled_mvi_static
from flexo_gradient_g3d import G3DMaterial


def setup():
    surface=NativeCoupledMVISurface(
        "CUF_TE1",G3DMaterial(80.,.22,.03,.003,.001,.8),
        grid=CoupledMVISurfaceGrid(9,5,1.3,2.),
        thickness_m=.2,potential_order=0,n_gauss=2)
    x=np.zeros((surface.n_fields,1,9))
    x[:,:,1:-1]=.04
    y=np.ones((surface.n_fields,1,5))
    return (surface,CoupledMVIFactors(x,y),
            CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","GROUNDED_BOTH"))


def test_electric_only_halfstep_does_not_modify_mechanical_fields(monkeypatch):
    surface,factors,program=setup()
    original=surface.reconstruct(factors)[:surface.n_mechanical]
    def fake_work(f,**kw):
        value=surface.reconstruct(f)
        node=np.zeros(surface.shape)
        c=surface.n_mechanical
        node[c]=value[c]*surface.grid.weights
        return CoupledMVIWork(
            .5*float(np.sum(value[c]*node[c])),
            value[:surface.n_mechanical],value[c:],
            np.einsum("cij,crj->cri",node,f.y),
            np.einsum("cij,cri->crj",node,f.x),
            f.rank,nodal_gradient=node)
    monkeypatch.setattr(surface,"weak_work",fake_work)
    result=solve_coupled_mvi_static(
        surface,factors,program=program,
        active_fields=(surface.n_mechanical,),
        field_scales=np.ones(surface.n_fields),energy_scale_j=1.)
    assert result.converged
    np.testing.assert_allclose(
        surface.reconstruct(result.factors)[:surface.n_mechanical],
        original,atol=2e-13,rtol=2e-13)


@pytest.mark.parametrize("invalid",[(0,0),(),(-1,),(99,),("0",)])
def test_invalid_active_field_set_rejected(invalid):
    surface,factors,program=setup()
    with pytest.raises((TypeError,ValueError),match="MVI_ACTIVE_FIELDS"):
        solve_coupled_mvi_static(
            surface,factors,program=program,active_fields=invalid,
            field_scales=np.ones(surface.n_fields),energy_scale_j=1.)
