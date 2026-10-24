#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference variational end/face BC for KL, FSDT, Reddy, and CUF TE1-TE4.

STATIC and DYNAMIC use exactly the SAME boundary functional. Mechanical end
supports and higher-order normal traces are explicit linear admissible-variation
constraints. Prescribed generalized dead end tractions, higher-order work,
springs and free electric charge contribute to ONE potential and its exact
discrete first variation. Electrode potential is an essential constraint.

This is a REFERENCE BC assembler. Physical microfree end closure, higher-order
natural boundary traces and stable SBP/PGD/FDM enforcement are not yet
independently verified. It does not run a shell PDE by itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import numpy as np

from cylinder_dynamic_loads import normalize_time_law

MECHANICAL_SUPPORTS=("FREE","CLAMPED","SIMPLY_SUPPORTED","GUIDED","SYMMETRY","CUSTOM")
HIGHER_TRACES=("NATURAL","FIX_FIRST_NORMAL","FIX_FIRST_AND_SECOND_NORMAL")
ELECTRIC_MODES=("INSULATED","POTENTIAL","CHARGE")
ENDS=("left","right")
ELECTRIC_SITES=("left","right","outer","inner")


def _number(value,label):
    if isinstance(value,bool):
        raise ValueError(f"{label} must be a finite numeric value")
    try:result=float(value)
    except (ValueError,TypeError) as exc:
        raise ValueError(f"{label} must be a finite numeric value") from exc
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(value,n,label):
    if value is None:return np.zeros(n)
    if not isinstance(value,(list,tuple,np.ndarray)):
        raise ValueError(f"{label}: expected exactly {n} finite entries")
    raw=np.asarray(value,dtype=object)
    if raw.shape!=(n,) or any(isinstance(x,(bool,np.bool_)) for x in raw.ravel()):
        raise ValueError(f"{label}: expected exactly {n} finite numeric entries")
    a=np.asarray(value,float)
    if a.shape!=(n,) or not np.isfinite(a).all():
        raise ValueError(f"{label}: expected exactly {n} finite entries")
    return a.copy()


def _object(parent,name):
    obj=parent.get(name,{})
    if not isinstance(obj,dict):
        raise ValueError(f"{name} must be a dictionary")
    return obj


def _electric_mode(data):
    mode=str(data.get("mode","INSULATED")).upper().strip()
    if mode not in ELECTRIC_MODES:
        raise ValueError(f"electric mode must be one of {ELECTRIC_MODES}")
    return mode


def _fixed_components(support,theory,n,site):
    # CUF unknowns in reference assembler: u0..uN,v0..vN,w0..wN.
    # DO NOT assume interleaved order u0,v0,w0,u1,v1,w1.
    cuf=theory.startswith("CUF_TE")
    m=n//3 if cuf else 1
    if support=="FREE":return ()
    if support=="CLAMPED":return tuple(range(n))
    if support=="SIMPLY_SUPPORTED":return (0,m,2*m)
    if support=="GUIDED":
        return tuple(range(m,3*m)) if cuf else ((1,2) if theory=="KL" else (1,2,3,4))
    if support=="SYMMETRY":
        return tuple(range(m)) if cuf else ((0,) if theory=="KL" else (0,3))
    selected=site.get("fixed_components",[])
    if not isinstance(selected,(tuple,list)) or any(type(i)!=int or i<0 or i>=n for i in selected):
        raise ValueError("CUSTOM fixed_components must be valid integer component IDs")
    if len(set(selected))!=len(selected):
        raise ValueError("duplicate CUSTOM component")
    return tuple(selected)


def _time_factor(data,mode,time_s):
    law=data.get("time_law")
    if law is None:return 1.
    parsed=normalize_time_law(law)
    if mode=="STATIC" and parsed.kind!="CONSTANT":
        raise ValueError("nonconstant time law is forbidden for STATIC boundaries")
    if mode=="DYNAMIC" and time_s is None:
        raise ValueError("DYNAMIC boundary load requires time_s")
    return float(parsed.value(0. if time_s is None else time_s))


@dataclass(frozen=True)
class ConstraintRow:
    field: str
    indices: tuple
    weights: tuple
    prescribed: float
    label: str

    def defect(self,mechanical,electrical):
        values=mechanical if self.field=="mechanical" else electrical
        if values is None:
            raise ValueError("electric constraints require electric unknowns")
        return float(np.dot(np.asarray(values).ravel()[list(self.indices)],self.weights)
                     -self.prescribed)


@dataclass(frozen=True)
class BoundaryResult:
    potential_correction_j: float
    mechanical_residual_correction: np.ndarray
    electrical_residual_correction: np.ndarray | None
    constraints: tuple
    gauge_required: bool
    free_electric_charge_c: float
    status: str="REFERENCE_BC_ONLY_GLOBAL_SOLVER_NOT_VERIFIED"
    absolute_free_electric_charge_c: float=0.

    def constraint_max_defect(self,mechanical,electrical):
        return max((abs(row.defect(mechanical,electrical))
                    for row in self.constraints),default=0.)


def boundary_passport(*,mechanical_left="FREE",mechanical_right="FREE",
                      higher_left="NATURAL",higher_right="NATURAL",
                      electric_left="INSULATED",electric_right="INSULATED",
                      electric_outer="INSULATED",electric_inner="INSULATED",
                      electric_values=None):
    """Create a deterministic, serializable mechanical+electrical BC passport."""
    mech={"left":{"support":mechanical_left,"higher":higher_left},
          "right":{"support":mechanical_right,"higher":higher_right}}
    values={} if electric_values is None else electric_values
    elec={}
    for site,mode in (("left",electric_left),("right",electric_right),
                      ("outer",electric_outer),("inner",electric_inner)):
        data={"mode":mode}
        if mode!="INSULATED":
            if site not in values:
                raise ValueError(f"electrical {site}: enter voltage or charge explicitly")
            data["value"]=_number(values[site],f"{site} electrical value")
        elec[site]=data
    return {"mechanical":mech,"electrical":elec}


def validate_boundary_passport(spec,*,electrical_enabled,calculation_kind="STATIC",
                               theory="CUF_TE1"):
    """Lightweight CASE/GUI validation without running shell calculations."""
    if not isinstance(spec,dict):
        raise ValueError("variational BC must be an object")
    kind=str(calculation_kind).strip().upper()
    if kind not in ("STATIC","DYNAMIC"):
        raise ValueError("calculation_kind must be STATIC/DYNAMIC")
    if theory not in ("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"):
        raise ValueError("Unknown kinematic theory for variational BC")
    if type(electrical_enabled) is not bool:
        raise ValueError("electrical_enabled must be boolean")
    mechanical=_object(spec,"mechanical")
    electrical=_object(spec,"electrical")
    n=(3 if theory=="KL" else 5 if theory in ("FSDT","REDDY")
       else 3*(int(theory[-1])+1))
    for side in ENDS:
        site=_object(mechanical,side)
        support=str(site.get("support","FREE")).upper().strip()
        high=str(site.get("higher","NATURAL")).upper().strip()
        if support not in MECHANICAL_SUPPORTS or high not in HIGHER_TRACES:
            raise ValueError(f"{side}: unsupported mechanical BC")
        if high=="FIX_FIRST_AND_SECOND_NORMAL" and theory.startswith("CUF_"):
            raise ValueError("second normal derivative trace unavailable in CUF_TE")
        if support=="CUSTOM" and "fixed_components" not in site:
            raise ValueError("CUSTOM needs explicit fixed_components")
        _fixed_components(support,theory,n,site)
        if support!="CUSTOM" and "fixed_components" in site:
            raise ValueError(side+": fixed_components requires CUSTOM")
        for key in ("prescribed","traction","double_traction","triple_traction",
                    "spring","spring_reference"):
            if key in site:
                values=_vector(site[key],n,side+"."+key)
                if key=="spring" and np.any(values<0):
                    raise ValueError(side+": spring must be nonnegative")
                if key=="triple_traction" and theory.startswith("CUF_") and np.any(values):
                    raise ValueError("triple_traction requires third surface jets")
        if "time_law" in site:
            parsed=normalize_time_law(site["time_law"])
            if kind=="STATIC" and parsed.kind!="CONSTANT":
                raise ValueError("STATIC cannot have time-dependent end loads")
    for side in ELECTRIC_SITES:
        site=_object(electrical,side)
        mode=_electric_mode(site)
        if mode=="INSULATED" and ("value" in site or "time_law" in site):
            raise ValueError(side+": INSULATED must not specify charge/voltage/time law")
        if not electrical_enabled and mode!="INSULATED":
            raise ValueError("electric BC requires electric model ON")
        if mode!="INSULATED":_number(site.get("value"),f"{side} voltage/charge")
        if "time_law" in site:
            parsed=normalize_time_law(site["time_law"])
            if kind=="STATIC" and parsed.kind!="CONSTANT":
                raise ValueError("STATIC cannot have time-dependent electric BC")
    for end in ENDS:
        a=_object(electrical,end)
        if _electric_mode(a)!="POTENTIAL":continue
        for face in ("outer","inner"):
            b=_object(electrical,face)
            if _electric_mode(b)!="POTENTIAL":continue
            # With time dependence this is only a local nominal check.
            # compile_boundary_program checks the effective voltage at t.
            if "time_law" not in a and "time_law" not in b:
                av=_number(a["value"],"end voltage")
                bv=_number(b["value"],"face voltage")
                if abs(av-bv)>1e-10*max(1,abs(av),abs(bv)):
                    raise ValueError("conflicting end/face electrode voltages")
    return True


def _row_weights(order,k,nx,ns,side,j,ops):
    i=0 if side=="left" else nx-1
    if order==0:axis=np.eye(nx)[i]
    elif order==1:axis=ops.Dx[i]
    else:axis=(ops.Dx@ops.Dx)[i]
    nz=np.flatnonzero(abs(axis)>1e-14)
    return (tuple((k*nx+int(a))*ns+j for a in nz),
            tuple(float(axis[a]) for a in nz))


def compile_boundary_program(functional,mechanical,electrical=None,*,spec=None,
                             calculation_kind="STATIC",time_s=None,
                             check_charge_balance=True):
    """Return exact additive first variations of conservative prescribed work.

    Constraints MUST be imposed by a later spatial MVI/FDM solver; here their
    rows are returned for inspection. Unspecified natural variations are free.
    """
    data={} if spec is None else spec
    validate_boundary_passport(
        data,electrical_enabled=bool(functional.electrical_on),
        calculation_kind=calculation_kind,theory=functional.theory)
    kind=str(calculation_kind).upper().strip()
    if kind=="DYNAMIC" and time_s is None:raise ValueError("DYNAMIC requires time_s")
    if time_s is not None:_number(time_s,"time_s")
    ops=functional.ops
    nx,ns=ops.shape;n=functional.n_mechanical
    q=np.asarray(mechanical,float)
    if q.shape!=(n,nx,ns) or not np.isfinite(q).all():
        raise ValueError("mechanical field shape mismatch")
    if functional.electrical_on:
        if electrical is None:raise ValueError("electric field required")
        phi=np.asarray(electrical,float)
        if (phi.ndim!=3 or phi.shape[1:]!=(nx,ns)
            or not 1<=phi.shape[0]<=5 or not np.isfinite(phi).all()):
            raise ValueError("electric field shape mismatch")
    else:
        if electrical is not None:raise ValueError("electric field disabled")
        phi=None
    mech=_object(data,"mechanical");electric=_object(data,"electrical")
    result_m=np.zeros_like(q)
    result_e=None if phi is None else np.zeros_like(phi)
    rows=[];energy=0.;total_charge=0.;absolute_charge=0.;has_electrode=False
    dx=ops.Dx;dxx=dx@dx

    def add(field,ids,weights,value,label):
        rows.append(ConstraintRow(field,ids,weights,_number(value,label),label))

    for side in ENDS:
        site=_object(mech,side)
        support=str(site.get("support","FREE")).upper().strip()
        higher=str(site.get("higher","NATURAL")).upper().strip()
        loc=0 if side=="left" else nx-1
        selected=_fixed_components(support,functional.theory,n,site)
        factor=_time_factor(site,kind,time_s)
        prescribed=factor*_vector(site.get("prescribed"),n,side+" prescribed")
        for j in range(ns):
            for k in selected:
                ids,w=_row_weights(0,k,nx,ns,side,j,ops)
                add("mechanical",ids,w,prescribed[k],f"{side} support q{k} s{j}")
            # Reference KL classical clamp includes zero normal slope.
            if functional.theory=="KL" and support in ("CLAMPED","GUIDED","SYMMETRY"):
                ids,w=_row_weights(1,2,nx,ns,side,j,ops)
                add("mechanical",ids,w,0.,f"{side} KL slope s{j}")
            if higher!="NATURAL":
                for k in range(n):
                    ids,w=_row_weights(1,k,nx,ns,side,j,ops)
                    add("mechanical",ids,w,0.,f"{side} micro d_x q{k} s{j}")
                    if higher=="FIX_FIRST_AND_SECOND_NORMAL":
                        ids,w=_row_weights(2,k,nx,ns,side,j,ops)
                        add("mechanical",ids,w,0.,f"{side} micro d_xx q{k} s{j}")

        traction=factor*_vector(site.get("traction"),n,side+" traction")
        double=factor*_vector(site.get("double_traction"),n,side+" double")
        triple=factor*_vector(site.get("triple_traction"),n,side+" triple")
        if functional.n_derivatives<10 and np.any(triple):
            raise ValueError("triple_traction requires third surface jets")
        spring=_vector(site.get("spring"),n,side+" spring")
        rest=_vector(site.get("spring_reference"),n,side+" reference")
        if np.any(spring<0):
            raise ValueError("negative spring stiffness")
        for j,wj in enumerate(ops.ws):
            v=q[:,loc,j]
            vx=dx[loc]@q[:,:,j].T
            vxx=dxx[loc]@q[:,:,j].T
            energy+=wj*(.5*float(np.dot(spring,(v-rest)**2))
                        -float(traction@v+double@vx+triple@vxx))
            result_m[:,loc,j]+=wj*(spring*(v-rest)-traction)
            result_m[:,:,j]-=wj*(double[:,None]*dx[loc][None,:]
                                 +triple[:,None]*dxx[loc][None,:])

    if phi is not None:
        p=phi.shape[0];R=float(functional.R);h=float(functional.h)
        # Axial end with constant-through-thickness V meets face electrode.
        for side in ENDS:
            end=_object(electric,side)
            if _electric_mode(end)!="POTENTIAL":continue
            ve=_number(end.get("value"),side+" voltage")*_time_factor(end,kind,time_s)
            for face in ("outer","inner"):
                face_site=_object(electric,face)
                if _electric_mode(face_site)!="POTENTIAL":continue
                vf=_number(face_site.get("value"),face+" voltage")*_time_factor(face_site,kind,time_s)
                if abs(ve-vf)>1e-10*max(1,abs(ve),abs(vf)):
                    raise ValueError("conflicting electrode voltages at "+side+"/"+face)
        for side in ELECTRIC_SITES:
            site=_object(electric,side)
            mode=_electric_mode(site)
            if mode=="INSULATED":continue
            value=_number(site.get("value"),side+" electric BC")*_time_factor(site,kind,time_s)
            if mode=="POTENTIAL":
                has_electrode=True
                if side in ENDS:
                    loc=0 if side=="left" else nx-1
                    for j in range(ns):
                        for k in range(p):
                            add("electrical",((k*nx+loc)*ns+j,),(1,),
                                value if k==0 else 0.,f"{side} electrode phi{k} s{j}")
                else:
                    parity=1 if side=="outer" else -1
                    basis=tuple(float(parity**k) for k in range(p))
                    for i in range(nx):
                        for j in range(ns):
                            indices=tuple((k*nx+i)*ns+j for k in range(p))
                            add("electrical",indices,basis,value,f"{side} electrode x{i} s{j}")
                continue
            if side in ENDS:
                loc=0 if side=="left" else nx-1
                z,w=np.polynomial.legendre.leggauss(max(8,2*p+2))
                z*=h/2;w*=h/2
                moments=np.array([np.sum(w*(1+z/R)*(2*z/h)**k) for k in range(p)])
                for j,wj in enumerate(ops.ws):
                    charge=wj*value*moments
                    energy-=float(charge@phi[:,loc,j])
                    result_e[:,loc,j]-=charge
                    total_charge+=float(charge[0])
                    absolute_charge+=abs(float(charge[0]))
            else:
                z=h/2 if side=="outer" else -h/2
                H=1+z/R
                basis=np.array([(2*z/h)**k for k in range(p)])
                for i in range(nx):
                    for j in range(ns):
                        charge=ops.wx[i]*ops.ws[j]*H*value*basis
                        energy-=float(charge@phi[:,i,j])
                        result_e[:,i,j]-=charge
                        total_charge+=float(charge[0])
                        absolute_charge+=abs(float(charge[0]))
        if check_charge_balance and not has_electrode and abs(total_charge)>1e-10*max(
            1.e-30,absolute_charge):
            raise ValueError("isolated electric system requires zero total free charge")

    # Duplicate trace rows can result from KL clamp plus gradient clamp.
    # Exact duplication is removed, contradictory prescriptions are rejected.
    unique={}
    for row in rows:
        key=(row.field,row.indices,row.weights)
        existing=unique.get(key)
        if existing is not None and abs(existing.prescribed-row.prescribed)>1e-10:
            raise ValueError("conflicting prescriptions for the same boundary trace")
        unique.setdefault(key,row)
    return BoundaryResult(float(energy),result_m,result_e,tuple(unique.values()),
                          phi is not None and not has_electrode,float(total_charge),
                          absolute_free_electric_charge_c=float(absolute_charge))


def combine_functional_and_boundary(functional,mechanical,electrical=None,*,bc_spec=None,
                                    dead_load=None,free_charge=None,
                                    calculation_kind="STATIC",time_s=None):
    """Stationary bulk enthalpy plus conservative end/face work (no PDE solve)."""
    bulk=functional.evaluate(mechanical,electrical,dead_load=dead_load,
                             free_charge=free_charge,energy_breakdown=False,
                             check_charge_balance=False)
    edge=compile_boundary_program(functional,mechanical,electrical,spec=bc_spec,
                                  calculation_kind=calculation_kind,time_s=time_s,
                                  check_charge_balance=False)
    if edge.gauge_required:
        volume=(np.zeros(functional.ops.shape) if free_charge is None else
                functional.W*np.asarray(free_charge,float)[0])
        net=float(np.sum(volume))+edge.free_electric_charge_c
        scale=float(np.sum(abs(volume)))+edge.absolute_free_electric_charge_c
        if abs(net)>1.e-10*max(1.e-30,scale):
            raise ValueError("isolated electric system requires zero total free charge (bulk + boundary)")
    rm=bulk.mechanical_residual+edge.mechanical_residual_correction
    re=(None if bulk.electrical_residual is None else
        bulk.electrical_residual+edge.electrical_residual_correction)
    return bulk.total_enthalpy_j+edge.potential_correction_j,rm,re,edge


__all__=("MECHANICAL_SUPPORTS","HIGHER_TRACES","ELECTRIC_MODES","ConstraintRow",
         "BoundaryResult","boundary_passport","validate_boundary_passport",
         "compile_boundary_program","combine_functional_and_boundary")

