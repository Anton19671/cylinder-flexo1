"""Synthetic physical CUF reconstruction and tabulated native MVI load contracts.

No continuum shell solve or time integration is executed by these tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from independent_mvi.coupled_load_history import (
    CoupledLoadProgram, CoupledTabulatedLoadLaw,
)
from independent_mvi.coupled_physical_recovery import (
    radial_history_row, recover_cuf_displacements,
)


def definition():
    return dict(kind="TABULATED_LINEAR",
                time_s=[0., .01, .02],
                mechanical_scale=[0., 1., 0.],
                electric_scale=[1., .5, 0.])


def test_three_stage_times_receive_interpolated_physical_sources():
    law=CoupledTabulatedLoadLaw.from_case(
        definition(),horizon_s=.02,electric_active=True)
    load=CoupledLoadProgram(
        np.full((3,9,5),2.),
        np.full((1,9,5),4.),
        np.full((1,9,5),.75),law)
    for time,mechanical_factor,electric_factor in (
            (0.,0.,1.),(.005,.5,.75),(.01,1.,.5),
            (.015,.5,.25),(.02,0.,0.)):
        mechanical,charge=load.at(time)
        np.testing.assert_allclose(mechanical,2*mechanical_factor)
        np.testing.assert_allclose(charge,4*electric_factor+.75)
    # A prescribed physical face charge cannot be incorrectly multiplied
    # by the time law of the volume free-charge source.
    np.testing.assert_allclose(load.at(.02)[1],.75)


def test_constant_load_program_remains_backward_compatible():
    loads=CoupledLoadProgram(np.ones((3,9,5)),np.ones((2,9,5))*2.)
    for time in (0.,.001,.5):
        mechanical,charge=loads.at(time)
        np.testing.assert_array_equal(mechanical,1.)
        np.testing.assert_array_equal(charge,2.)


@pytest.mark.parametrize("broken",(
    dict(kind="TABULATED_LINEAR",time_s=[.001,.02],
         mechanical_scale=[0.,1.]),
    dict(kind="TABULATED_LINEAR",time_s=[0.,0.],
         mechanical_scale=[0.,1.]),
    dict(kind="TABULATED_LINEAR",time_s=[0.,.01],
         mechanical_scale=[0.,1.]),
    dict(kind="TABULATED_LINEAR",time_s=[0.,.02],
         mechanical_scale=[0.,float("nan")]),
    dict(kind="TABULATED_LINEAR",time_s=[0.,.02],
         mechanical_scale=[0.]),
    dict(kind="UNKNOWN",time_s=[0.,.02],
         mechanical_scale=[0.,1.]),
))
def test_invalid_time_laws_fail_closed(broken):
    with pytest.raises(ValueError):
        CoupledTabulatedLoadLaw.from_case(
            broken,horizon_s=.02,electric_active=False)


def test_electric_load_law_forbidden_when_electricity_off():
    with pytest.raises(ValueError,match="ELECTRIC_SCALE_WITH_ELECTRICITY_OFF"):
        CoupledTabulatedLoadLaw.from_case(
            definition(),horizon_s=.02,electric_active=False)


def test_no_extrapolation_of_loaded_stage():
    law=CoupledTabulatedLoadLaw.from_case(
        definition(),horizon_s=.02,electric_active=True)
    for invalid in (-1., .03, float("nan"),float("inf"),True):
        with pytest.raises(ValueError,match="OUTSIDE_TABULATION"):
            law.factors_at(invalid)


class _Grid:
    nx=9
    ns=5


class _Surface:
    def __init__(self,theory,thickness=.2):
        self.theory=theory
        self.thickness_m=thickness
        self.grid=_Grid()
        self.n_mechanical=(3*(int(theory[-1])+1)
                           if theory.startswith("CUF") else 3)


@pytest.mark.parametrize("theory",("CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"))
def test_three_physical_cuf_faces_reconstruct_normalized_basis(theory):
    surface=_Surface(theory)
    n=int(theory[-1])+1
    field=np.zeros((3*n,9,5))
    # U=1+2*zeta, V=3*zeta^2, W=1e-6*(1-zeta+zeta^3...)
    field[0]=1.
    field[1]=2.
    if n>2:field[n+2]=3.
    for k in range(n):
        field[2*n+k]=(-1.)**k *1.e-6
    got=recover_cuf_displacements(surface,field)
    assert got.displacement_stations_m.shape==(3,3,9,5)
    np.testing.assert_allclose(
        got.z_stations_m,[-.1,0.,.1])
    for z_index,zeta in enumerate((-1.,0.,1.)):
        np.testing.assert_allclose(got.displacement_stations_m[z_index,0],
                                   1.+2*zeta)
        expected_w=sum(((-1.)**k)*1.e-6*zeta**k for k in range(n))
        np.testing.assert_allclose(
            got.displacement_stations_m[z_index,2],expected_w)
    summary=radial_history_row(surface,field,step=3,time_s=.015)
    assert summary["step"]==3
    assert summary["axial_node"]==4
    assert summary["circumferential_node"]==0
    assert summary["middle_w_m"]==pytest.approx(1.e-6)
    assert not got.displacement_stations_m.flags.writeable


def test_classical_theory_is_not_mislabeled_as_cuf_surface_polynomial():
    surface=_Surface("KL")
    field=np.zeros((3,9,5))
    field[2]=4.e-6
    assert recover_cuf_displacements(surface,field) is None
    summary=radial_history_row(surface,field,step=1,time_s=.01)
    assert summary["middle_w_m"]==pytest.approx(4.e-6)
    assert summary["inner_w_m"]==""
    assert summary["outer_w_m"]==""
