"""Independent energy-consistent cylindrical FDM, all seven local shell laws.

The differential operators here are derived from MKRFieldJets itself,
not imported from MVI/SBP/reference variational solvers. Its residual is
the exact first variation of the trapezoidally integrated *native FDM*
kinematic enthalpy (and, when supplied, inertia/viscosity weak work).
It is a distinct EXPERIMENTAL alternative to the native strong operator;
the latter is not silently changed.

Physical natural end data are external virtual work. Essential end data
are imposed as exact linear constraints by a small per-component nullspace,
NOT by penalty or arbitrary replacement of strong rows. This accommodates
physical normal traces through second order at BOTH cylinder ends.
Electric insulated gauges may be imposed as one exact global affine
constraint, not a single arbitrary substituted electric equation.

No source/MVI import, no solver dispatch, no automatic physical time steps,
and no assertion of continuum or mesh convergence.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np

from cylinder_direct_strong_fdm import DirectStrongGrid
from cylinder_mkr_field_jets import MKRFieldJets
from cylinder_direct_strong_boundary import _condition
from cylinder_direct_strong_model import DirectStrongShell
from cylinder_direct_strong_newton import (
    StrongElectricGauge, StrongNewtonSettings,
    solve_strong_nonlinear_equations,
)


@lru_cache(maxsize=24)
def _stencil_matrices(nx, ns, length_m, radius_m):
    """Return actual field-jet matrices, with no independent re-stencilling."""
    grid=DirectStrongGrid(nx,ns,length_m,radius_m)
    jets=MKRFieldJets(grid)
    # A full tensor product of axial/angular identity probes would
    # allocate O(nx**2*ns + ns**2*nx) field values. For experimental
    # fine meshes this can exhaust RAM *before* the first Newton call.
    # Probe exactly the SAME native differential operator in bounded
    # blocks instead; the final derivative matrices use O(nx**2+ns**2).
    ax=[np.eye(nx)]+[np.empty((nx,nx),dtype=float) for _ in (1,2,3)]
    az=[np.eye(ns)]+[np.empty((ns,ns),dtype=float) for _ in (1,2,3)]
    block=8
    for start in range(0,nx,block):
        stop=min(nx,start+block)
        xb=np.zeros((stop-start,nx,ns),dtype=float)
        for local,index in enumerate(range(start,stop)):
            xb[local,index,:]=1.
        for order in (1,2,3):
            ax[order][:,start:stop]=(
                jets.derivative(xb,order,0)[:,:,0].T)
    for start in range(0,ns,block):
        stop=min(ns,start+block)
        sb=np.zeros((stop-start,nx,ns),dtype=float)
        for local,index in enumerate(range(start,stop)):
            sb[local,:,index]=1.
        for order in (1,2,3):
            az[order][:,start:stop]=(
                jets.derivative(sb,0,order)[:,0,:].T)
    matrices=tuple(tuple(mats) for mats in (ax,az))
    for item in matrices:
        for matrix in item:
            matrix.setflags(write=False)
    return matrices


def _apply_adjoint(coefficients, Dx, Ds):
    return np.einsum("cij,jk->cik",
        np.einsum("ij,cjk->cik",Dx.T,coefficients,optimize=True),
        Ds,optimize=True)


@dataclass(frozen=True)
class NativeEnergyWork:
    mechanical: np.ndarray
    electrical: np.ndarray | None
    status: str = "EXACT_NATIVE_DISCRETE_FIRST_VARIATION_NOT_MESH_VERIFIED"


class NativeEnergyFDM:
    """Native-FDM weighted first variation of the actual seven-theory jets."""

    def __init__(self, model: DirectStrongShell):
        if not isinstance(model,DirectStrongShell):
            raise TypeError("NativeEnergyFDM requires native DirectStrongShell")
        if model.grid.axial_scheme!="LEGACY_TRAPEZOID":
            raise ValueError("NATIVE_ENERGY_FDM_REQUIRES_LEGACY_GRID_SCHEME")
        self.model=model
        self.axial,self.angular=_stencil_matrices(
            model.grid.nx,model.grid.ns,
            model.grid.length_m,model.grid.radius_m)
        self.weight=model.area_weights
        if not (np.isfinite(self.weight).all() and np.all(self.weight>0)):
            raise ValueError("NATIVE_ENERGY_FDM_WEIGHTS_INVALID")

    def jet_action(self, mechanical_coefficients,
                   electrical_coefficients=None, *,
                   mechanical_load=None, electrical_charge=None):
        """Integrate jet coefficients by *exact* transpose of actual jets."""
        model=self.model
        m=np.asarray(mechanical_coefficients,float)
        if (m.shape!=(len(model.derivative_orders),*model.shape)
                or not np.isfinite(m).all()):
            raise ValueError("NATIVE_ENERGY_FDM_MECHANICAL_JET_SHAPE")
        out=np.zeros(model.shape)
        for k,(_,ax,az) in enumerate(model.derivative_orders):
            out+=_apply_adjoint(m[k]*self.weight[None,:,:],
                                self.axial[ax],self.angular[az])
        if mechanical_load is not None:
            f=np.asarray(mechanical_load,float)
            if f.shape!=model.shape or not np.isfinite(f).all():
                raise ValueError("NATIVE_ENERGY_FDM_LOAD_INVALID")
            out-=self.weight[None,:,:]*f

        electrical=None
        if model.electrical_on:
            e=np.asarray(electrical_coefficients,float)
            if (e.ndim!=4 or e.shape[0]!=3 or
                    e.shape[2:]!=model.shape[1:] or e.shape[1]<1
                    or e.shape[1]>5 or not np.isfinite(e).all()):
                raise ValueError("NATIVE_ENERGY_FDM_ELECTRIC_JET_SHAPE")
            electrical=np.zeros_like(e[0])
            for i,(ax,az) in enumerate(((0,0),(1,0),(0,1))):
                electrical+=_apply_adjoint(e[i]*self.weight[None,:,:],
                                            self.axial[ax],self.angular[az])
            if electrical_charge is not None:
                charge=np.asarray(electrical_charge,float)
                if charge.shape!=electrical.shape or not np.isfinite(charge).all():
                    raise ValueError("NATIVE_ENERGY_FDM_CHARGE_INVALID")
                electrical+=self.weight[None,:,:]*charge
        elif electrical_coefficients is not None or electrical_charge is not None:
            raise ValueError("NATIVE_ENERGY_FDM_ELECTRICITY_OFF")
        return NativeEnergyWork(out,electrical)

    def balance(self, mechanical, electrical=None, *,
                mechanical_load=None, electrical_charge=None,
                velocity=None, acceleration=None, density_kg_m3=None,
                eta_s=0., mass_damping_rate_per_s=0.):
        """Full native enthalpy/dynamic virtual work, without BC elimination."""
        mc,ec=self.model.coefficients(
            mechanical,electrical,velocity=velocity,
            acceleration=acceleration,density_kg_m3=density_kg_m3,
            eta_s=eta_s,mass_damping_rate_per_s=mass_damping_rate_per_s)
        return self.jet_action(
            mc,ec,mechanical_load=mechanical_load,
            electrical_charge=electrical_charge)


@dataclass(frozen=True)
class _FieldSpace:
    base: np.ndarray
    null: np.ndarray
    external_work: np.ndarray

    @property
    def nfree(self):
        return self.null.shape[1]


def _field_space(grid, axial, ends, field, *, mechanical):
    """SVD elimination of true kinematic essential traces at both ends."""
    n=grid.nx
    ns=grid.ns
    constraints=[]
    values=[]
    external=np.zeros((n,ns))
    for side_index,side in enumerate(("left","right")):
        for order,entry in enumerate(ends[side][field:field+1][0]):
            if not mechanical and order:
                raise ValueError("ELECTRIC_END_ONLY_POTENTIAL_OR_FLUX")
            kind,value=_condition(
                entry,ns,f"native energy {side} field {field} order {order}")
            normal=(-1. if side=="left" and order%2 else 1.)
            row=normal*axial[order][0 if side_index==0 else -1,:]
            if kind=="ESSENTIAL":
                constraints.append(row.copy())
                values.append(value)
            else:
                # External force/potential natural work, both faces.
                external+=grid.ds*row[:,None]*value[None,:]

    if not constraints:
        return _FieldSpace(np.zeros((n,ns)),np.eye(n),external)
    C=np.stack(constraints)
    target=np.stack(values)
    # Row normalization preserves exact constraints while comparing
    # slope/curvature rows with different derivative units.
    norms=np.linalg.norm(C,axis=1)
    if not np.isfinite(norms).all() or np.any(norms<=0):
        raise ValueError("NATIVE_ENERGY_DEGENERATE_ESSENTIAL_TRACE")
    scaled=C/norms[:,None]
    target=target/norms[:,None]
    U,s,Vh=np.linalg.svd(scaled,full_matrices=True)
    tolerance=512.*np.finfo(float).eps*max(scaled.shape)*s[0]
    if np.count_nonzero(s>tolerance)!=len(constraints):
        raise ValueError("NATIVE_ENERGY_DEPENDENT_ESSENTIAL_TRACES")
    base=Vh[:len(s)].T@((U.T@target)/s[:,None])
    null=Vh[len(s):].T.copy()
    defect=np.max(np.abs(scaled@base-target))
    allowed=1.e-9*max(1.,float(np.max(np.abs(target))))
    if defect>allowed:
        raise ValueError("NATIVE_ENERGY_INCOMPATIBLE_ESSENTIAL_TRACES")
    null.setflags(write=False)
    return _FieldSpace(base,null,external)


class NativeEnergyEndSpace:
    """Exact constrained trial/test space for physical mechanical/electric ends.

    A natural condition is EXTERNAL virtual work; an essential condition
    is eliminated in the native FDM kinematic basis. A supplied floating
    potential gauge is imposed as a global affine constraint by removing
    one reduced electric coordinate; no unphysical penalization occurs.
    """

    def __init__(self, engine: NativeEnergyFDM, *,
                 axial_orders, mechanical_ends, electrical_ends=None,
                 electric_gauge=None):
        if not isinstance(engine,NativeEnergyFDM):
            raise TypeError("NativeEnergyFDM required")
        self.engine=engine
        model=engine.model
        m=tuple(axial_orders)
        if (len(m)!=model.n_mechanical or
                any(type(v) is not int or v not in (1,2,3)
                    for v in m)):
            raise ValueError("EXPLICIT_AXIAL_HALF_ORDERS_REQUIRED")
        if not isinstance(mechanical_ends,dict) or set(mechanical_ends)!={"left","right"}:
            raise ValueError("EXPLICIT_MECHANICAL_LEFT_RIGHT_ENDS_REQUIRED")
        self.mechanical_spaces=[]
        for c,order in enumerate(m):
            for side in ("left","right"):
                if (len(mechanical_ends[side])!=model.n_mechanical or
                        len(mechanical_ends[side][c])!=order):
                    raise ValueError("MKR_ENERGY_END_COUNT_MISMATCH")
            self.mechanical_spaces.append(
                _field_space(model.grid,engine.axial,mechanical_ends,c,
                             mechanical=True))
        self.electrical_spaces=[]
        self.electric_gauge=electric_gauge
        self.insulated_constant_mode=False
        if model.electrical_on:
            if not isinstance(electrical_ends,dict) or set(electrical_ends)!={"left","right"}:
                raise ValueError("EXPLICIT_ELECTRIC_LEFT_RIGHT_ENDS_REQUIRED")
            count=len(electrical_ends["left"])
            if (count<1 or count>5 or len(electrical_ends["right"])!=count or
                any(not isinstance(electrical_ends[s][i],dict)
                    for s in ("left","right") for i in range(count))):
                raise ValueError("NATIVE_ENERGY_ELECTRIC_END_COUNT_MISMATCH")
            wrapped={
                side:[[electrical_ends[side][i]] for i in range(count)]
                for side in ("left","right")}
            for i in range(count):
                self.electrical_spaces.append(
                    _field_space(model.grid,engine.axial,wrapped,i,
                                 mechanical=False))
            all_nat=all(electrical_ends[s][0]["kind"]=="NATURAL"
                        for s in ("left","right"))
            self.insulated_constant_mode=all_nat
            if not all_nat and electric_gauge is not None:
                raise ValueError("NATIVE_ENERGY_GAUGE_WITH_PRESCRIBED_POTENTIAL")
            if all_nat and electric_gauge is None:
                raise ValueError("UNGAUGED_INSULATED_ELECTRICAL_MODE")
        elif electrical_ends is not None or electric_gauge is not None:
            raise ValueError("NATIVE_ENERGY_ELECTRICITY_OFF")
        self.spaces=tuple(self.mechanical_spaces+self.electrical_spaces)
        self.nmech=len(self.mechanical_spaces)
        self.ns=model.grid.ns
        self.counts=tuple(space.nfree*self.ns for space in self.spaces)
        self.free_size=sum(self.counts)
        if self.free_size<1:
            raise ValueError("NATIVE_ENERGY_ALL_VARIABLES_FIXED")
        self.gauge_row=None
        self.gauge_pivot=None
        self.gauge_rhs=None
        if electric_gauge is not None:
            if (not isinstance(electric_gauge,StrongElectricGauge) or
                    electric_gauge.weights.shape!=(
                        len(self.electrical_spaces),model.grid.nx,self.ns)):
                raise ValueError("NATIVE_ENERGY_GAUGE_SHAPE_INVALID")
            g=np.zeros(self.free_size)
            base_value=0.
            offset=sum(self.counts[:self.nmech])
            for space,w in zip(self.electrical_spaces,electric_gauge.weights):
                base_value+=float(np.sum(space.base*w))
                n=space.nfree*self.ns
                g[offset:offset+n]=(space.null.T@w).ravel()
                offset+=n
            pivot=int(np.argmax(np.abs(g)))
            if abs(g[pivot])<=1.e-13*max(1.,float(np.max(np.abs(g)))):
                raise ValueError("NATIVE_ENERGY_GAUGE_INCOMPATIBLE_ESSENTIALS")
            self.gauge_row=g
            self.gauge_pivot=pivot
            self.gauge_rhs=float(electric_gauge.target_v-base_value)
        self.reduced_size=self.free_size-int(self.gauge_pivot is not None)
        if self.reduced_size<1:
            raise ValueError("NATIVE_ENERGY_ZERO_REDUCED_DOF")

    def check_electric_source(self,charge):
        """A gauge fixes the potential reference, never an unbalanced source.

        The constant virtual-potential direction requires total volume/face
        charge to equal the prescribed outward axial electric end work.
        Eliminating its coordinate must not hide an incompatible Neumann
        problem. Test source work before any Newton or material evaluation.
        """
        if charge is not None:
            value=np.asarray(charge,float)
            shape=(len(self.electrical_spaces),*self.engine.model.shape[1:])
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError("NATIVE_ENERGY_CHARGE_INVALID")
        else:
            value=None
        if not self.insulated_constant_mode:
            return
        source=(np.zeros_like(self.engine.weight) if value is None else
                self.engine.weight*value[0])
        external=self.electrical_spaces[0].external_work
        # Scale before summation to retain dimensional extremes.
        scale=max(float(np.max(np.abs(source))),float(np.max(np.abs(external))))
        if scale==0.:
            return
        source=source/scale
        external=external/scale
        net=math.fsum(source.ravel())-math.fsum(external.ravel())
        total=math.fsum(np.abs(source).ravel())+math.fsum(np.abs(external).ravel())
        if abs(net)>1.e-10*total:
            raise ValueError("NATIVE_ENERGY_INCOMPATIBLE_INSULATED_CHARGE")

    def _full_from_reduced(self,free):
        vector=np.asarray(free,float).ravel()
        if vector.shape!=(self.reduced_size,) or not np.isfinite(vector).all():
            raise ValueError("NATIVE_ENERGY_REDUCED_STATE_INVALID")
        if self.gauge_pivot is None:
            return vector
        all_free=np.empty(self.free_size)
        other=np.arange(self.free_size)!=self.gauge_pivot
        all_free[other]=vector
        all_free[self.gauge_pivot]=(
            self.gauge_rhs-float(self.gauge_row[other]@vector)
            )/self.gauge_row[self.gauge_pivot]
        if not np.isfinite(all_free).all():
            raise ValueError("NATIVE_ENERGY_GAUGE_RECONSTRUCTION_INVALID")
        return all_free

    def decode(self,free):
        all_free=self._full_from_reduced(free)
        fields=[]
        offset=0
        for space,n in zip(self.spaces,self.counts):
            y=all_free[offset:offset+n].reshape(space.nfree,self.ns)
            fields.append(space.base+space.null@y)
            offset+=n
        mech=np.stack(fields[:self.nmech])
        elec=(np.stack(fields[self.nmech:])
              if self.electrical_spaces else None)
        return mech,elec

    def encode(self,mechanical,electrical=None):
        model=self.engine.model
        q=np.asarray(mechanical,float)
        if q.shape!=model.shape or not np.isfinite(q).all():
            raise ValueError("NATIVE_ENERGY_MECHANICAL_GUESS_INVALID")
        fields=list(q)
        if self.electrical_spaces:
            p=np.asarray(electrical,float)
            if p.shape!=(len(self.electrical_spaces),*model.shape[1:]) or not np.isfinite(p).all():
                raise ValueError("NATIVE_ENERGY_ELECTRIC_GUESS_INVALID")
            fields.extend(list(p))
        elif electrical is not None:
            raise ValueError("NATIVE_ENERGY_ELECTRICITY_OFF")
        free=np.concatenate([
            (space.null.T@(field-space.base)).ravel()
            for space,field in zip(self.spaces,fields)])
        if self.gauge_pivot is not None:
            free=np.delete(free,self.gauge_pivot)
        return free

    def residual(self,work: NativeEnergyWork):
        q=np.asarray(work.mechanical,float)
        if q.shape!=self.engine.model.shape or not np.isfinite(q).all():
            raise ValueError("NATIVE_ENERGY_MECHANICAL_RESIDUAL_INVALID")
        fields=list(q)
        if self.electrical_spaces:
            p=np.asarray(work.electrical,float)
            if p.shape!=(len(self.electrical_spaces),*q.shape[1:]) or not np.isfinite(p).all():
                raise ValueError("NATIVE_ENERGY_ELECTRIC_RESIDUAL_INVALID")
            fields.extend(list(p))
        elif work.electrical is not None:
            raise ValueError("NATIVE_ENERGY_UNEXPECTED_ELECTRIC_RESIDUAL")
        full=np.concatenate([
            (space.null.T@(field-space.external_work)).ravel()
            for space,field in zip(self.spaces,fields)])
        if self.gauge_pivot is None:
            return full
        other=np.arange(self.free_size)!=self.gauge_pivot
        return (full[other]-
            (self.gauge_row[other]/self.gauge_row[self.gauge_pivot])
            *full[self.gauge_pivot])

    def solve_static(self, mechanical_guess, electrical_guess=None, *,
                     mechanical_load=None, electrical_charge=None,
                     reduced_state_scale, reduced_equation_scale,
                     settings=StrongNewtonSettings()):
        """Solve only static constrained native FDM; no time evolution.

        Reduced scales must be supplied in their true mixed mechanical/
        electric dimensions. No source rank, auto order, or gauge is guessed.
        """
        self.check_electric_source(electrical_charge)
        guess=self.encode(mechanical_guess,electrical_guess)
        packed=guess.reshape(1,1,-1)
        def equations(y,_unused):
            q,p=self.decode(y.ravel())
            work=self.engine.balance(q,p,mechanical_load=mechanical_load,
                                     electrical_charge=electrical_charge)
            return self.residual(work).reshape(1,1,-1),None
        def packed_scale(value,label):
            # StrongNewton accepts a scalar or a tensor with the EXACT
            # (component,nx,ns) shape. This reduced chart packs all
            # independent degrees of freedom as (1,1,nfree).
            scale=np.asarray(value,float)
            if scale.ndim==0:
                return float(scale)
            if scale.shape!=(self.reduced_size,):
                raise ValueError(label+" requires scalar or reduced free size")
            return scale.reshape(1,1,self.reduced_size)

        newton=solve_strong_nonlinear_equations(
            equations,packed,
            mechanical_state_scale=packed_scale(
                reduced_state_scale,"NATIVE_ENERGY_STATE_SCALE"),
            mechanical_equation_scale=packed_scale(
                reduced_equation_scale,"NATIVE_ENERGY_EQUATION_SCALE"),
            settings=settings)
        if not newton.converged:
            return NativeEnergySolution(None,None,newton)
        q,p=self.decode(newton.mechanical.ravel())
        return NativeEnergySolution(q,p,newton)


@dataclass(frozen=True)
class NativeEnergySolution:
    mechanical: np.ndarray | None
    electrical: np.ndarray | None
    nonlinear: object
    status: str = "EXPERIMENTAL_NATIVE_ENERGY_FDM_NOT_MESH_CERTIFIED"


__all__=("NativeEnergyFDM","NativeEnergyWork","NativeEnergyEndSpace",
         "NativeEnergySolution")
