"""No physical shell solves: global insulated MVI electric gauge contracts.

The physical gauge is a single additive constant in psi_0, independent
of the rank-R separated factors. Tests verify exact curved-shell volume
mean, source compatibility, essential axial traces, and the precisely
representable null direction in reduced X/Y coordinates. No MKR, SVD,
full-grid displacement Newton, or loaded cylinder simulation runs.
"""
import json

import numpy as np
import pytest

from flexo_gradient_g3d import G3DMaterial
from independent_mvi.coupled_surface import (
    CoupledMVISurfaceGrid,CoupledMVIFactors,NativeCoupledMVISurface,
)
from independent_mvi.coupled_halfstep import (
    CoupledMVIEndProgram,normalized_mvi_factors,
    _floating_reduced_null,_check_end_factors,
)
from independent_mvi import run_coupled_research as runner
from test_independent_coupled_mvi_case_dispatch import make_case


def make_surface(order=2):
    surf=NativeCoupledMVISurface(
        "CUF_TE2",G3DMaterial(100.,.24,.02,0.,0.,1.),
        grid=CoupledMVISurfaceGrid(11,7,1.4,2.),
        thickness_m=.2,potential_order=order,n_gauss=2,
        floating_gauge=True)
    x=np.zeros((surf.n_fields,1,surf.grid.nx))
    y=np.ones((surf.n_fields,1,surf.grid.ns))
    c=surf.n_mechanical
    x[c,0]=np.linspace(-.7,.4,surf.grid.nx)
    if order>=1:
        x[c+1,0]=.2
    if order>=2:
        x[c+2,0]=-.3
    return surf,CoupledMVIFactors(x,y)


@pytest.mark.parametrize("order",range(5))
def test_single_floating_offset_produces_zero_physical_volume_mean(order):
    surf,seed=make_surface(order)
    raw=np.einsum("cri,crj->cij",seed.x,seed.y)
    assert abs(surf._floating_offset(raw))>1.e-6
    assert abs(surf.floating_gauge_integral(seed))<2.e-14
    field=surf.reconstruct(seed)
    np.testing.assert_allclose(
        field[surf.n_mechanical+1:],
        raw[surf.n_mechanical+1:],atol=0.,rtol=0.)
    correction=field[surf.n_mechanical]-raw[surf.n_mechanical]
    np.testing.assert_allclose(correction,correction[0,0],
                               rtol=0.,atol=2.e-15)


def test_floating_electric_program_leaves_both_axial_ends_free():
    surf,seed=make_surface(1)
    program=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS",
                                 "INSULATED_FLOATING")
    free=_check_end_factors(surf,seed,program)
    for c in range(surf.n_mechanical,surf.n_fields):
        np.testing.assert_array_equal(free[c],np.arange(surf.grid.nx))
    for c in range(surf.n_mechanical):
        np.testing.assert_array_equal(free[c],np.arange(1,surf.grid.nx-1))


@pytest.mark.parametrize("axis",("X","Y"))
def test_only_exact_representable_constant_null_direction_is_removed(axis):
    surf,seed=make_surface(0)
    c=surf.n_mechanical
    prog=CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS","INSULATED_FLOATING")
    working=normalized_mvi_factors(surf,seed,prog,axis)
    fields=tuple(range(surf.n_fields))
    free=prog.free_axial_indices(surf)
    sizes=[working.rank*(len(free[c]) if axis=="X" else surf.grid.ns)
           for c in fields]
    indices=(free if axis=="X" else
             tuple(np.arange(surf.grid.ns) for _ in fields))
    direction=_floating_reduced_null(
        surf,working,axis,fields,indices,sizes,np.ones(surf.n_fields))
    if axis=="X":
        assert direction is not None
        changed_x=working.x.copy()
        changed_y=working.y.copy()
        offset=sum(sizes[:c])
        changed_x[c,:,free[c]]+=direction[offset:offset+sizes[c]].reshape(
            working.rank,len(free[c]))
        shifted=CoupledMVIFactors(changed_x,changed_y)
        np.testing.assert_allclose(
            surf.reconstruct(working),surf.reconstruct(shifted),
            atol=2.e-13,rtol=2.e-13)
    else:
        # The axial profile was deliberately nonconstant; the global
        # constant shift cannot be represented by changing only Y.
        assert direction is None


def test_charged_floating_problem_rejected_before_solver(tmp_path):
    case,doc=make_case(tmp_path,electric=True)
    doc["boundary"]["electric"]="INSULATED_FLOATING"
    baseline=np.load(tmp_path/"loads.npz")
    mechanical=np.array(baseline["mechanical_load"],copy=True)
    baseline.close()
    np.savez_compressed(tmp_path/"loads.npz",
                        mechanical_load=mechanical,
                        free_charge=np.ones((1,9,5))*.1)
    case.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="INCOMPATIBLE_TOTAL_FREE_CHARGE"):
        runner.load_case(case)
    assert not (tmp_path/"output").exists()


def test_balanced_floating_charge_case_passes_preflight(tmp_path):
    case,doc=make_case(tmp_path,electric=True)
    doc["boundary"]["electric"]="INSULATED_FLOATING"
    case.write_text(json.dumps(doc))
    result=runner.run(case,validate_only=True)
    assert result["electric_floating_gauge"] is True
    assert result["physical_steps_computed"]==0
    assert not (tmp_path/"output").exists()


def test_floating_gauge_rejects_grounded_program():
    surf,seed=make_surface()
    with pytest.raises(ValueError,match="REQUIRES_INSULATED_ENDS"):
        CoupledMVIEndProgram("IMMOBILE_PARALLEL_ENDS",
                             "GROUNDED_LEFT").validate(surf)
