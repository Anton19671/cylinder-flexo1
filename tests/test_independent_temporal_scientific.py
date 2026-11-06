"""Manufactured single-mode temporal audit of six independent integration routes.

Exercise the REAL Newmark, Bathe and two-stage Gauss-IRK4 method bodies from
both independent solver classes, replacing only their spatial constitutive
operators with one exactly known harmonic oscillator:

    q'' + omega0**2 q = amplitude * sin(omega_drive*t).

This is NOT a shell computation or physical validation of a spatial method;
it tests their time-update and staged algebra against closed-form dynamics.
"""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from independent_mkr.linear_models import IndependentStrongModel
from independent_mvi.linear_models import IndependentVariationalModel


OMEGA0=3.7
DRIVE_HZ=.91
DRIVE_OMEGA=2*np.pi*DRIVE_HZ
AMPLITUDE=2.3
END_TIME=.64


class ScalarStrong(IndependentStrongModel):
    def __init__(self, amplitude=AMPLITUDE, omega=OMEGA0):
        self.case=SimpleNamespace(nx=1,ns=1,frequency_hz=DRIVE_HZ)
        self.n=1
        self.dofs=(1,)
        self.basis=(np.ones((1,1)),)
        self.omega_sq=omega**2
        self.force=np.full((1,1,1),amplitude)

    def mass_action(self,q):
        return np.asarray(q,float).copy()

    def stiffness_action(self,q):
        return self.omega_sq*np.asarray(q,float)


class ScalarVariational(IndependentVariationalModel):
    def __init__(self, amplitude=AMPLITUDE, omega=OMEGA0):
        self.case=SimpleNamespace(nx=1,ns=1,frequency_hz=DRIVE_HZ)
        self.n=1
        self.rank=1
        self.dofs=(1,)
        self.basis=(np.ones((1,1)),)
        self.omega_sq=omega**2
        self.area=np.ones((1,1))
        self.force=np.full((1,1,1),amplitude)

    def mass_action(self,q):
        return np.asarray(q,float).copy()

    def stiffness_action(self,q):
        return self.omega_sq*np.asarray(q,float)


def _state(q,v,a):
    return tuple(np.array([[[x]]],dtype=float) for x in (q,v,a))


def _scalar(v):
    return float(np.asarray(v).reshape(-1)[0])


def _exact_forced(time):
    denominator=OMEGA0**2-DRIVE_OMEGA**2
    q=(AMPLITUDE/denominator)*(
        np.sin(DRIVE_OMEGA*time)
        -(DRIVE_OMEGA/OMEGA0)*np.sin(OMEGA0*time))
    v=(AMPLITUDE*DRIVE_OMEGA/denominator)*(
        np.cos(DRIVE_OMEGA*time)-np.cos(OMEGA0*time))
    return q,v


def _advance(model,scheme,state,now,dt,factors):
    if isinstance(model,ScalarStrong):
        if scheme=="NEWMARK":
            return model.step_newmark(*state,now,dt),None
        if scheme=="BATHE":
            _,end=model.step_bathe(*state,now,dt)
            return end,None
        _,end=model.step_gauss_irk4(*state,now,dt)
        return end,None
    if scheme=="NEWMARK":
        end,factors,_=model.step_newmark(*state,now,dt,previous=factors)
    elif scheme=="BATHE":
        _,end,factors,_=model.step_bathe(*state,now,dt,previous=factors)
    else:
        _,end,factors,_=model.step_gauss_irk4(
            *state,now,dt,previous=factors)
    return end,factors


def integrate(model_type,scheme,dt,*,free_vibration=False,omega=OMEGA0):
    amplitude=0. if free_vibration else AMPLITUDE
    model=model_type(amplitude=amplitude,omega=omega)
    state=_state(1.,0.,-omega**2) if free_vibration else _state(0.,0.,0.)
    steps=round(END_TIME/dt)
    assert abs(steps*dt-END_TIME)<1e-12
    factors=None
    energy=[.5*(_scalar(state[1])**2+omega**2*_scalar(state[0])**2)]
    for i in range(steps):
        state,factors=_advance(model,scheme,state,i*dt,dt,factors)
        q,v,a=map(_scalar,state)
        assert np.isfinite([q,v,a]).all()
        # Independently known forced oscillator equation at END of step.
        driving=0. if free_vibration else AMPLITUDE*np.sin(
            DRIVE_OMEGA*(i+1)*dt)
        np.testing.assert_allclose(
            a+omega**2*q,driving,rtol=3e-8,atol=3e-8)
        energy.append(.5*(v**2+omega**2*q**2))
    return state,np.asarray(energy)


@pytest.mark.parametrize("model_type",(ScalarStrong,ScalarVariational))
@pytest.mark.parametrize("scheme,min_ratio",(
    ("NEWMARK",2.8),("BATHE",2.6),("GAUSS_IRK4",10.0),
))
def test_manufactured_forced_oscillator_time_refinement(
    model_type,scheme,min_ratio
):
    coarse,_=integrate(model_type,scheme,.08)
    medium,_=integrate(model_type,scheme,.04)
    fine,_=integrate(model_type,scheme,.02)
    correct_q,correct_v=_exact_forced(END_TIME)
    errors=[]
    for state in (coarse,medium,fine):
        q,v,_=map(_scalar,state)
        errors.append(np.hypot(q-correct_q,(v-correct_v)/OMEGA0))
    assert errors[0]>errors[1]>errors[2]
    assert errors[0]/errors[1]>=min_ratio
    assert errors[1]/errors[2]>=min_ratio


@pytest.mark.parametrize("model_type",(ScalarStrong,ScalarVariational))
@pytest.mark.parametrize("scheme",("NEWMARK","GAUSS_IRK4"))
def test_undamped_oscillator_energy_is_conserved_to_roundoff(model_type,scheme):
    _,energies=integrate(model_type,scheme,.04,free_vibration=True)
    np.testing.assert_allclose(energies,energies[0],rtol=3e-8,atol=3e-8)


@pytest.mark.parametrize("model_type",(ScalarStrong,ScalarVariational))
def test_bathe_no_external_load_has_nonincreasing_oscillator_energy(model_type):
    # Bathe may dissipate numerical high-frequency energy, unlike symplectic Gauss.
    _,energies=integrate(model_type,"BATHE",.04,
                         free_vibration=True,omega=12.)
    assert np.max(np.diff(energies))<=2e-8*energies[0]
    assert energies[-1]<energies[0]


@pytest.mark.parametrize("model_type",(ScalarStrong,ScalarVariational))
@pytest.mark.parametrize("scheme",("NEWMARK","BATHE","GAUSS_IRK4"))
def test_invalid_time_step_rejected_before_any_spatial_solve(model_type,scheme):
    solver=model_type()
    state=_state(0.,0.,0.)
    func=dict(NEWMARK=solver.step_newmark,BATHE=solver.step_bathe,
              GAUSS_IRK4=solver.step_gauss_irk4)[scheme]
    for dt in (0.,-1.,float("nan"),float("inf")):
        with pytest.raises(ValueError):
            func(*state,0.,dt)
