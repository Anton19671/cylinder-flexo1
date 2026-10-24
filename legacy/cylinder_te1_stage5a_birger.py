#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUF-TE1 closed-cylinder Stage 5A:
one-load-step outer Birger material cycle.

Source: supplied verified Word derivation, Appendix G.9, equations (728)-(741).

The material diagram Phi(epsilon_i) is an EXPLICIT input callable.
No material curve, interpolation or extrapolation rule is embedded here.

V-5a audit implementation:
1) solve once at the initial frozen GB^0;
2) at outer iteration k, evaluate material at the already converged pair
   (state^k, GB^k);
3) calculate GB_calc^{k+1} and relax to GB^{k+1};
4) solve once at the UPDATED frozen GB^{k+1};
5) evaluate the final material criteria at that same pair
   (state^{k+1}, GB^{k+1});
6) if not accepted, reuse that converged pair directly as the starting pair of
   the next outer iteration instead of solving GB^{k+1} a second time.

No displacement field obtained for an old GB can be stored as converged together
with a new GB.  V-5 only removes the redundant repeated inner solve.  Reviewer
V-6 additionally requires an independent outer convergence check on the
reconstructed displacement/generalized-displacement field (eta_u).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
import final_progress as fprog

import cylinder_te1_local_kernel as nk
import cylinder_te1_stage2b as s2
import cylinder_te1_stage3a_connector as c3
import cylinder_te1_stage3c_internal_mvi as c4


class BirgerOuterError(RuntimeError):
    pass


class MaterialDiagramError(BirgerOuterError):
    pass


class InnerSolveNotConverged(BirgerOuterError):
    pass


class BirgerAdmissibilityError(BirgerOuterError):
    """The trial material state lies outside deformation-theory validity."""

    def __init__(self, code, message):
        self.code=str(code)
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True)
class BirgerSettings:
    epsilon_tol: float
    omega: float
    G_ref: float
    sigma_ref: float
    tol_G: float
    tol_sigma: float
    tol_u: float
    max_outer_iterations: int
    epsilon_i_limit: float
    delta_J_limit: float
    # K5 (audit): the material diagram is sigma_i = Phi(epsilon_i), so its
    # initial elastic slope must be 3*G0.  A small explicit tolerance permits
    # tabulated/experimental round-off without silently accepting an ordinary
    # uniaxial sigma-epsilon curve.
    initial_slope_rtol: float = 0.05
    reject_local_unloading: bool = True
    unloading_tolerance: float = 1.0e-12
    # REJECT is the physical default: a material-validity limit makes the
    # load step unacceptable.  FLAG preserves legacy exploratory workflows
    # only when requested explicitly in the CASE.
    limit_policy: str = "REJECT"

    def __post_init__(self):
        if self.epsilon_tol < 0.0:
            raise ValueError("epsilon_tol must be >=0")
        if not (0.0 < self.omega <= 1.0):
            raise ValueError("omega must satisfy 0<omega<=1")
        if self.G_ref <= 0.0:
            raise ValueError("G_ref must be >0")
        if self.sigma_ref <= 0.0:
            raise ValueError("sigma_ref must be >0")
        if self.tol_G <= 0.0 or self.tol_sigma <= 0.0 or self.tol_u <= 0.0:
            raise ValueError("Birger tolerances tol_G, tol_sigma and tol_u must be >0")
        if self.max_outer_iterations < 1:
            raise ValueError("max_outer_iterations must be >=1")
        if self.epsilon_i_limit <= 0.0:
            raise ValueError("epsilon_i_limit must be explicitly >0")
        if self.delta_J_limit <= 0.0:
            raise ValueError("delta_J_limit must be explicitly >0")
        if str(self.limit_policy).upper() not in ("REJECT","FLAG"):
            raise ValueError("limit_policy must be REJECT or FLAG")
        if not np.isfinite(self.initial_slope_rtol) or not (0.0 < self.initial_slope_rtol <= 0.5):
            raise ValueError("initial_slope_rtol must satisfy 0 < value <= 0.5")
        if not isinstance(self.reject_local_unloading,(bool,np.bool_)):
            raise ValueError("reject_local_unloading must be boolean")
        if (not np.isfinite(self.unloading_tolerance)
                or self.unloading_tolerance < 0.0):
            raise ValueError("unloading_tolerance must be finite and >=0")


@dataclass(frozen=True)
class MaterialPointState:
    epsilon_i: np.ndarray
    sigma_i: np.ndarray
    detF: np.ndarray

    def __post_init__(self):
        e=np.asarray(self.epsilon_i,float)
        s=np.asarray(self.sigma_i,float)
        j=np.asarray(self.detF,float)
        if e.shape != s.shape or e.shape != j.shape:
            raise ValueError("epsilon_i, sigma_i, detF shape mismatch")
        if not (np.all(np.isfinite(e)) and np.all(np.isfinite(s)) and np.all(np.isfinite(j))):
            raise ValueError("non-finite material-point state")
        if np.any(j <= 0.0):
            raise s2.InvalidDeformationMapping(
                f"INVALID_DEFORMATION_MAPPING min_detF={float(np.min(j)):.17g}"
            )
        if np.any(e < 0.0) or np.any(s < 0.0):
            raise ValueError("epsilon_i and sigma_i must be non-negative")
        object.__setattr__(self,"epsilon_i",e)
        object.__setattr__(self,"sigma_i",s)
        object.__setattr__(self,"detF",j)


@dataclass
class BirgerLoadStepResult:
    state: object
    GB: np.ndarray
    converged: bool
    status: str
    outer_iterations: int
    history: list
    material_state: MaterialPointState
    unloading_mask: Optional[np.ndarray]
    max_epsilon_i: float
    max_abs_J_minus_1: float
    epsilon_limit_exceeded: bool
    delta_J_limit_exceeded: bool
    eta_u: float
    critical_epsilon_i_index: Optional[tuple] = None
    critical_J_index: Optional[tuple] = None
    unloading_point_fraction: float = 0.0


def _flat_index_tuple(a, flat_index):
    return tuple(int(i) for i in np.unravel_index(int(flat_index),a.shape))


def material_admissibility(mat,settings,previous_accepted_epsilon_i=None):
    """Evaluate the hard Birger validity criteria at every material point."""
    eps=np.asarray(mat.epsilon_i,float)
    det=np.asarray(mat.detF,float)
    abs_dj=np.abs(det-1.0)
    ie=int(np.argmax(eps)); ij=int(np.argmax(abs_dj))
    max_eps=float(eps.reshape(-1)[ie])
    max_dj=float(abs_dj.reshape(-1)[ij])
    eps_index=_flat_index_tuple(eps,ie)
    j_index=_flat_index_tuple(det,ij)

    # These are validity limits, not output-only warnings.  Only round-off at
    # the exact threshold is ignored.
    eps_guard=64.0*np.finfo(float).eps*max(1.0,abs(settings.epsilon_i_limit))
    j_guard=64.0*np.finfo(float).eps*max(1.0,abs(settings.delta_J_limit))
    eps_exceeded=bool(max_eps > settings.epsilon_i_limit+eps_guard)
    dj_exceeded=bool(max_dj > settings.delta_J_limit+j_guard)

    unloading_mask=None
    unloading_fraction=0.0
    min_delta=0.0
    unloading_index=None
    if previous_accepted_epsilon_i is not None:
        prev=np.asarray(previous_accepted_epsilon_i,float)
        if prev.shape != eps.shape:
            raise ValueError("previous_accepted_epsilon_i shape mismatch")
        scale=max(1.0,float(np.max(np.abs(prev))),float(np.max(np.abs(eps))))
        tol=max(float(settings.unloading_tolerance),64.0*np.finfo(float).eps*scale)
        delta=eps-prev
        unloading_mask=delta < -tol
        if np.any(unloading_mask):
            unloading_fraction=float(np.mean(unloading_mask))
            iu=int(np.argmin(delta))
            min_delta=float(delta.reshape(-1)[iu])
            unloading_index=_flat_index_tuple(delta,iu)

    return {
        "max_epsilon_i":max_eps,
        "max_abs_J_minus_1":max_dj,
        "epsilon_limit_exceeded":eps_exceeded,
        "delta_J_limit_exceeded":dj_exceeded,
        "critical_epsilon_i_index":eps_index,
        "critical_J_index":j_index,
        "unloading_mask":unloading_mask,
        "unloading_point_fraction":unloading_fraction,
        "min_delta_epsilon_i":min_delta,
        "unloading_index":unloading_index,
    }


def require_material_admissible(mat,settings,previous_accepted_epsilon_i=None):
    """Raise a coded failure so the load driver can reduce/reject the step."""
    d=material_admissibility(mat,settings,previous_accepted_epsilon_i)
    if str(settings.limit_policy).upper()=="REJECT":
        if d["epsilon_limit_exceeded"]:
            raise BirgerAdmissibilityError(
                "BIRGER_EPSILON_I_LIMIT_EXCEEDED",
                f"max_epsilon_i={d['max_epsilon_i']:.17g} > "
                f"epsilon_i_limit={settings.epsilon_i_limit:.17g}; "
                f"material_point_index={d['critical_epsilon_i_index']}"
            )
        if d["delta_J_limit_exceeded"]:
            raise BirgerAdmissibilityError(
                "BIRGER_VOLUME_CHANGE_LIMIT_EXCEEDED",
                f"max_abs_J_minus_1={d['max_abs_J_minus_1']:.17g} > "
                f"delta_J_limit={settings.delta_J_limit:.17g}; "
                f"material_point_index={d['critical_J_index']}"
            )
    if (settings.reject_local_unloading
            and d["unloading_point_fraction"] > 0.0):
        raise BirgerAdmissibilityError(
            "BIRGER_LOCAL_UNLOADING_NOT_SUPPORTED",
            f"fraction={d['unloading_point_fraction']:.17g}; "
            f"min_delta_epsilon_i={d['min_delta_epsilon_i']:.17g}; "
            f"material_point_index={d['unloading_index']}; "
            f"tolerance={settings.unloading_tolerance:.17g}"
        )
    return d


MaterialDiagram = Callable[[np.ndarray], np.ndarray]
InnerSolve = Callable[[object, np.ndarray], object]
MaterialEvaluator = Callable[[object, np.ndarray], MaterialPointState]
StateVector = Callable[[object], np.ndarray]


def _diagram_values(diagram: MaterialDiagram, epsilon_i):
    eps=np.asarray(epsilon_i,float)
    phi=np.asarray(diagram(eps),float)
    if phi.ndim == 0:
        phi=np.full_like(eps,float(phi))
    else:
        try:
            phi=np.broadcast_to(phi,eps.shape).astype(float,copy=True)
        except ValueError as exc:
            raise MaterialDiagramError(
                f"material diagram output shape {phi.shape} cannot broadcast to {eps.shape}"
            ) from exc
    if not np.all(np.isfinite(phi)):
        raise MaterialDiagramError("material diagram returned non-finite values")
    if np.any(phi < 0.0):
        raise MaterialDiagramError("material diagram returned negative stress intensity")
    return phi




def validate_material_diagram_monotonic(
    material_diagram: MaterialDiagram, *, epsilon_i_limit: float,
    samples: int = 4097,
):
    """V10: numerically reject any descending branch Phi'(epsilon_i)<0.

    Hardening and plateaus are allowed.  The user curve is never smoothed or
    modified.  TABLE/CSV points are additionally checked exactly by
    final_birger_curve._clean_points; this numerical guard protects formula,
    piecewise and external-Python callables and direct launcher use.
    """
    limit=float(epsilon_i_limit); samples=int(samples)
    if not np.isfinite(limit) or limit<=0.0:
        raise ValueError("epsilon_i_limit must be finite and >0")
    if samples<3:
        raise ValueError("V10 samples must be >=3")
    eps=np.linspace(0.0,limit,samples,dtype=float)
    try:
        phi=_diagram_values(material_diagram,eps)
    except Exception as exc:
        raise MaterialDiagramError(
            f"V10 Birger curve cannot be evaluated on [0,{limit:.17g}]: {exc}"
        ) from exc
    d=np.diff(phi)
    tol=128.0*np.finfo(float).eps*max(1.0,float(np.max(np.abs(phi))))
    bad=np.flatnonzero(d < -tol)
    if bad.size:
        i=int(bad[0])
        raise MaterialDiagramError(
            "V10 Birger curve must be nondecreasing on the declared range: "
            f"Phi({eps[i+1]:.17g})={phi[i+1]:.17g} < "
            f"Phi({eps[i]:.17g})={phi[i]:.17g}"
        )
    return {
        "epsilon_i_limit":limit,
        "samples":samples,
        "min_increment":float(np.min(d)) if d.size else 0.0,
        "status":"V10_PASS",
    }


def validate_material_diagram_near_origin(
    material_diagram: MaterialDiagram, *, G0: float, epsilon_tol: float,
    initial_slope_rtol: float = 0.05, probe_epsilon: float = 1.0e-8,
):
    """K5/K6 audit guard for the Birger input diagram.

    The verified Birger law uses Phi(epsilon_i)=sigma_i and therefore

        dPhi/depsilon_i |_(0+) = 3 G0.

    This check prevents two dangerous input mistakes without changing the
    constitutive equations:
      * K5: passing an ordinary uniaxial sigma-epsilon curve;
      * K6: a curve that does not pass through the origin, which can make
            Phi/(3*epsilon_i) explode immediately above epsilon_tol.

    Returns diagnostics for GUI/self-check logging.
    """
    G0=float(G0); epsilon_tol=float(epsilon_tol)
    rtol=float(initial_slope_rtol); h=float(probe_epsilon)
    if not np.isfinite(G0) or G0 <= 0.0:
        raise ValueError("G0 must be finite and positive")
    if not np.isfinite(epsilon_tol) or epsilon_tol < 0.0:
        raise ValueError("epsilon_tol must be finite and >=0")
    if not np.isfinite(rtol) or not (0.0 < rtol <= 0.5):
        raise ValueError("initial_slope_rtol must satisfy 0 < value <= 0.5")
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("probe_epsilon must be finite and positive")

    # Always probe the mathematical curve itself near zero.  Strain is
    # dimensionless; 1e-8 is far below ordinary yield strains and is still
    # numerically well resolved in both MPa and Pa unit systems.
    eps_probe=np.array([0.0,h],float)
    try:
        phi=_diagram_values(material_diagram,eps_probe)
    except Exception as exc:
        raise MaterialDiagramError(
            "Birger curve must be defined from epsilon_i=0 and in a small "
            f"right neighbourhood of zero: {exc}"
        ) from exc

    expected_slope=3.0*G0
    # Scale the origin tolerance to the expected stress at the probe strain.
    origin_atol=max(100.0*np.finfo(float).eps*G0, expected_slope*h*1.0e-6)
    phi0=float(phi[0]); phih=float(phi[1])
    if abs(phi0) > origin_atol:
        raise MaterialDiagramError(
            "K6 Birger curve invalid near origin: Phi(0) must be 0 for the "
            "sigma_i=Phi(epsilon_i) diagram; "
            f"got Phi(0)={phi0:.17g}. Add/define the origin point (0,0)."
        )

    slope=(phih-phi0)/h
    if not np.isfinite(slope) or slope <= 0.0:
        raise MaterialDiagramError(
            "Birger curve initial slope must be finite and positive"
        )
    rel=abs(slope-expected_slope)/expected_slope
    if rel > rtol:
        raise MaterialDiagramError(
            "K5 Birger curve initial elastic slope is inconsistent with "
            "sigma_i=Phi(epsilon_i): expected dPhi/depsilon_i=3*G0="
            f"{expected_slope:.17g}, got {slope:.17g} "
            f"(relative mismatch {rel:.3%}, allowed {rtol:.3%}). "
            "Do not pass an ordinary uniaxial sigma-epsilon curve without "
            "conversion to the intensity diagram sigma_i-epsilon_i."
        )

    # K6 second guard: immediately above the numerical epsilon_tol, the secant
    # modulus must not jump above the initial elastic modulus by more than the
    # same declared input tolerance.  This catches hidden HOLD/interpolation
    # plateaus even when Phi(0) itself was special-cased by a user callable.
    onset=max(h, np.nextafter(epsilon_tol, np.inf))
    if onset <= epsilon_tol:
        onset=max(h, epsilon_tol*(1.0+1.0e-8)+h)
    try:
        phi_on=float(_diagram_values(material_diagram,np.array([onset],float))[0])
    except Exception as exc:
        raise MaterialDiagramError(
            "Birger curve is not defined at the active secant-modulus onset "
            f"epsilon_i={onset:.17g}: {exc}"
        ) from exc
    G_on=phi_on/(3.0*onset)
    if not np.isfinite(G_on) or G_on <= 0.0:
        raise MaterialDiagramError(
            "Birger curve gives non-positive/non-finite secant shear modulus "
            f"at epsilon_i={onset:.17g}"
        )
    if G_on > G0*(1.0+rtol):
        raise MaterialDiagramError(
            "K6 Birger curve creates an excessive secant modulus immediately "
            "above epsilon_tol: GB/G0="
            f"{G_on/G0:.6g} > {1.0+rtol:.6g}. Check the origin, epsilon_tol "
            "and interpolation/extrapolation of Phi(epsilon_i)."
        )

    return {
        "phi0":phi0,
        "probe_epsilon":h,
        "initial_slope":slope,
        "expected_initial_slope":expected_slope,
        "initial_slope_relative_error":rel,
        "active_onset_epsilon":onset,
        "active_onset_GB_over_G0":G_on/G0,
    }


def birger_secant_field(epsilon_i, phi_sigma, *, G0, epsilon_tol):
    """
    Word equations (730), equivalently (381)-(382):
      GB_calc = Phi(epsilon_i)/(3 epsilon_i), epsilon_i > epsilon_tol
      GB_calc = G0,                         epsilon_i <= epsilon_tol
    """
    eps=np.asarray(epsilon_i,float)
    phi=np.asarray(phi_sigma,float)
    if eps.shape != phi.shape:
        raise ValueError("epsilon_i/phi_sigma shape mismatch")
    if G0 <= 0.0:
        raise ValueError("G0 must be positive")
    out=np.full_like(eps,float(G0))
    active=eps > float(epsilon_tol)
    out[active]=phi[active]/(3.0*eps[active])
    if not np.all(np.isfinite(out)) or np.any(out <= 0.0):
        raise MaterialDiagramError(
            "calculated Birger secant shear modulus must be finite and positive"
        )
    return out


def relaxed_GB(GB_old, GB_calc, omega):
    """
    Word equation (731):
      GB^{k+1} = (1-omega) GB^k + omega GB_calc^{k+1}
    """
    a=np.asarray(GB_old,float)
    b=np.asarray(GB_calc,float)
    if a.shape != b.shape:
        raise ValueError("GB_old/GB_calc shape mismatch")
    out=(1.0-float(omega))*a+float(omega)*b
    if np.any(out <= 0.0) or not np.all(np.isfinite(out)):
        raise BirgerOuterError("relaxed GB is non-positive or non-finite")
    return out


def eta_G_pre(GB_new, GB_old, *, G0, G_ref):
    """
    Word equation (732).
    """
    return float(np.max(np.abs(np.asarray(GB_new)-np.asarray(GB_old))/(float(G0)+float(G_ref))))


def eta_sigma(sigma_i, phi_sigma, *, sigma_ref):
    """
    Word equations (733) and (737).
    """
    return float(np.max(np.abs(np.asarray(sigma_i)-np.asarray(phi_sigma))/float(sigma_ref)))


def eta_G_final(GB_check, GB_new, *, G0, G_ref):
    """
    Word equation (736).
    """
    return float(np.max(np.abs(np.asarray(GB_check)-np.asarray(GB_new))/(float(G0)+float(G_ref))))


def solve_birger_outer_cycle(
    initial_state,
    initial_GB,
    *,
    G0: float,
    material_diagram: MaterialDiagram,
    inner_solve: InnerSolve,
    evaluate_material: MaterialEvaluator,
    state_vector: StateVector,
    settings: BirgerSettings,
    previous_accepted_epsilon_i=None,
):
    """
    Generic one-load-step outer Birger cycle.

    The inner solver, material evaluator and physical-state vectorizer are
    injected explicitly so the outer algorithm can be tested independently and
    then used with MVI profiles or full-grid FDM unknowns without comparing raw
    separated X/Y factors.
    """
    G0=float(G0)
    if G0 <= 0.0:
        raise ValueError("G0 must be positive")

    # K5/K6 audit guard.  Validation only rejects inconsistent input curves;
    # it does not alter Phi, GB or any converged state.
    validate_material_diagram_near_origin(
        material_diagram, G0=G0, epsilon_tol=settings.epsilon_tol,
        initial_slope_rtol=settings.initial_slope_rtol,
    )
    validate_material_diagram_monotonic(
        material_diagram,epsilon_i_limit=settings.epsilon_i_limit
    )

    GBk=np.asarray(initial_GB,float).copy()
    if np.any(GBk <= 0.0) or not np.all(np.isfinite(GBk)):
        raise ValueError("initial_GB must be finite and positive")

    history=[]

    # Keep the relaxation coefficient exactly as specified by the case/GUI.
    # The Birger method here is a fixed secant outer iteration; changing omega
    # during the run changes the submitted material iteration.
    omega_k=float(settings.omega)

    # V-5a: establish one fully consistent initial pair (state^0, GB^0).
    # On later outer iterations this pair is reused from the previous updated
    # solve instead of solving the same frozen GB a second time.
    inner_at_GBk=inner_solve(initial_state,GBk.copy())
    if not getattr(inner_at_GBk,"converged",False):
        raise InnerSolveNotConverged(
            "INNER_MVI_NOT_CONVERGED_AT_INITIAL_GB / outer=1"
        )

    for k in range(settings.max_outer_iterations):
        old_state=inner_at_GBk.state

        # 1. Material state at the already converged current pair (state^k, GB^k).
        mat_old=evaluate_material(old_state,GBk.copy())
        admissibility_old=material_admissibility(
            mat_old,settings,previous_accepted_epsilon_i
        )
        try:
            require_material_admissible(
                mat_old,settings,previous_accepted_epsilon_i
            )
        except BirgerAdmissibilityError as exc:
            history.append({
                "outer_iteration":k+1,
                "status":exc.code,
                **{key:value for key,value in admissibility_old.items()
                   if key != "unloading_mask"},
            })
            return BirgerLoadStepResult(
                state=old_state,
                GB=GBk.copy(),
                converged=False,
                status=str(exc),
                outer_iterations=k+1,
                history=history,
                material_state=mat_old,
                unloading_mask=admissibility_old["unloading_mask"],
                max_epsilon_i=admissibility_old["max_epsilon_i"],
                max_abs_J_minus_1=admissibility_old["max_abs_J_minus_1"],
                epsilon_limit_exceeded=admissibility_old["epsilon_limit_exceeded"],
                delta_J_limit_exceeded=admissibility_old["delta_J_limit_exceeded"],
                eta_u=0.0,
                critical_epsilon_i_index=admissibility_old["critical_epsilon_i_index"],
                critical_J_index=admissibility_old["critical_J_index"],
                unloading_point_fraction=admissibility_old["unloading_point_fraction"],
            )
        phi_old=_diagram_values(material_diagram,mat_old.epsilon_i)

        # 2. Trial secant field and 3. relaxation.
        GBcalc=birger_secant_field(
            mat_old.epsilon_i,phi_old,G0=G0,epsilon_tol=settings.epsilon_tol
        )
        GBnew=relaxed_GB(GBk,GBcalc,omega_k)

        # 4. Preliminary diagnostics. They NEVER accept the step by themselves.
        etaGpre=eta_G_pre(GBnew,GBk,G0=G0,G_ref=settings.G_ref)
        etaSpre=eta_sigma(mat_old.sigma_i,phi_old,sigma_ref=settings.sigma_ref)

        # 5. ONE inner solve at UPDATED GB^{k+1}.  If the step is not accepted,
        # this exact result becomes inner_at_GBk on the next outer iteration.
        inner_at_GBnew=inner_solve(old_state,GBnew.copy())
        if not getattr(inner_at_GBnew,"converged",False):
            raise InnerSolveNotConverged(
                f"FINAL_INNER_MVI_NOT_CONVERGED_AT_UPDATED_GB / outer={k+1}"
            )
        final_state=inner_at_GBnew.state

        # Reviewer V-6: eta_sigma is algebraically tied to eta_G for the
        # Birger secant law.  Compare the actual reconstructed displacement /
        # generalized-displacement field between outer iterations instead.
        U_old=np.asarray(state_vector(old_state),float).reshape(-1)
        U_new=np.asarray(state_vector(final_state),float).reshape(-1)
        if U_old.shape != U_new.shape:
            raise BirgerOuterError("V6 state_vector shape changed between outer iterations")
        if not (np.all(np.isfinite(U_old)) and np.all(np.isfinite(U_new))):
            raise BirgerOuterError("V6 state_vector returned non-finite values")
        etaUfin=float(np.linalg.norm(U_new-U_old)/max(np.linalg.norm(U_new),1.0e-300))

        # 7. Re-evaluate at exactly the same updated GB used for final_state.
        mat_fin=evaluate_material(final_state,GBnew.copy())
        admissibility=material_admissibility(
            mat_fin,settings,previous_accepted_epsilon_i
        )
        try:
            require_material_admissible(
                mat_fin,settings,previous_accepted_epsilon_i
            )
        except BirgerAdmissibilityError as exc:
            history.append({
                "outer_iteration":k+1,
                "status":exc.code,
                "eta_u":etaUfin,
                **{key:value for key,value in admissibility.items()
                   if key != "unloading_mask"},
            })
            return BirgerLoadStepResult(
                state=final_state,
                GB=GBnew.copy(),
                converged=False,
                status=str(exc),
                outer_iterations=k+1,
                history=history,
                material_state=mat_fin,
                unloading_mask=admissibility["unloading_mask"],
                max_epsilon_i=admissibility["max_epsilon_i"],
                max_abs_J_minus_1=admissibility["max_abs_J_minus_1"],
                epsilon_limit_exceeded=admissibility["epsilon_limit_exceeded"],
                delta_J_limit_exceeded=admissibility["delta_J_limit_exceeded"],
                eta_u=etaUfin,
                critical_epsilon_i_index=admissibility["critical_epsilon_i_index"],
                critical_J_index=admissibility["critical_J_index"],
                unloading_point_fraction=admissibility["unloading_point_fraction"],
            )
        phi_fin=_diagram_values(material_diagram,mat_fin.epsilon_i)
        GBcheck=birger_secant_field(
            mat_fin.epsilon_i,phi_fin,G0=G0,epsilon_tol=settings.epsilon_tol
        )
        etaGfin=eta_G_final(GBcheck,GBnew,G0=G0,G_ref=settings.G_ref)
        etaSfin=eta_sigma(mat_fin.sigma_i,phi_fin,sigma_ref=settings.sigma_ref)

        max_eps=admissibility["max_epsilon_i"]
        max_dJ=admissibility["max_abs_J_minus_1"]
        eps_exceeded=admissibility["epsilon_limit_exceeded"]
        dJ_exceeded=admissibility["delta_J_limit_exceeded"]

        history.append({
            "outer_iteration":k+1,
            "old_inner_status":getattr(inner_at_GBk,"status","UNKNOWN"),
            "final_inner_status":getattr(inner_at_GBnew,"status","UNKNOWN"),
            "eta_G_pre":etaGpre,
            "eta_sigma_pre":etaSpre,
            "eta_G_fin":etaGfin,
            "eta_sigma_fin":etaSfin,
            "eta_u":etaUfin,
            "max_epsilon_i":max_eps,
            "max_abs_J_minus_1":max_dJ,
            "epsilon_limit_exceeded":eps_exceeded,
            "delta_J_limit_exceeded":dJ_exceeded,
            "critical_epsilon_i_index":admissibility["critical_epsilon_i_index"],
            "critical_J_index":admissibility["critical_J_index"],
            "unloading_point_fraction":admissibility["unloading_point_fraction"],
        })
        fprog.emit("birger_outer",
            outer_iteration=int(k+1),
            eta_G_pre=float(etaGpre), eta_sigma_pre=float(etaSpre),
            eta_G=float(etaGfin), eta_sigma=float(etaSfin), eta_u=float(etaUfin),
            GB_ratio_min=float(np.min(GBnew)/float(G0)),
            GB_ratio_max=float(np.max(GBnew)/float(G0)),
            max_epsilon_i=float(max_eps),
            max_abs_J_minus_1=float(max_dJ),
            stage="BIRGER_OUTER")

        # Reviewer V-6: sigma_i == 3*GB*epsilon_i identically for the
        # adopted Birger secant constitutive form, so eta_sigma is not an
        # independent convergence measure.  eta_u is independent and catches
        # a converged GB field while the displacement field is still drifting.
        # Keep eta_sigma as requested in the audit AFTER block and add eta_u.
        # 8. Acceptance only after final solve + final material/field criteria.
        if (etaGfin <= settings.tol_G
                and etaSfin <= settings.tol_sigma
                and etaUfin <= settings.tol_u):
            unloading_mask=admissibility["unloading_mask"]

            return BirgerLoadStepResult(
                state=final_state,
                GB=GBnew.copy(),
                converged=True,
                status="CONVERGED_BIRGER",
                outer_iterations=k+1,
                history=history,
                material_state=mat_fin,
                unloading_mask=unloading_mask,
                max_epsilon_i=max_eps,
                max_abs_J_minus_1=max_dJ,
                epsilon_limit_exceeded=eps_exceeded,
                delta_J_limit_exceeded=dJ_exceeded,
                eta_u=etaUfin,
                critical_epsilon_i_index=admissibility["critical_epsilon_i_index"],
                critical_J_index=admissibility["critical_J_index"],
                unloading_point_fraction=admissibility["unloading_point_fraction"],
            )

        # 9. Word equation (739): GB^k <- GB^{k+1}; continue outer cycle.
        # V-5a: reuse the already converged state at exactly that GB.
        GBk=GBnew
        inner_at_GBk=inner_at_GBnew

    # Return the last fully consistent pair, but explicitly NOT accepted.
    return BirgerLoadStepResult(
        state=final_state,
        GB=GBnew.copy(),
        converged=False,
        status="MAX_BIRGER_OUTER_ITERATIONS",
        outer_iterations=settings.max_outer_iterations,
        history=history,
        material_state=mat_fin,
        unloading_mask=None,
        max_epsilon_i=max_eps,
        max_abs_J_minus_1=max_dJ,
        epsilon_limit_exceeded=eps_exceeded,
        delta_J_limit_exceeded=dJ_exceeded,
        eta_u=etaUfin,
        critical_epsilon_i_index=admissibility["critical_epsilon_i_index"],
        critical_J_index=admissibility["critical_J_index"],
        unloading_point_fraction=admissibility["unloading_point_fraction"],
    )


def _GB_value(GB, g, idx, nz, surf_shape):
    a=np.asarray(GB,float)
    if a.ndim == 0:
        return float(a)
    if a.shape == surf_shape:
        return float(a[idx])
    if a.shape == (nz,)+surf_shape:
        return float(a[(g,)+idx])
    raise ValueError(
        f"GB must be scalar, surface-shaped {surf_shape}, or {(nz,)+surf_shape}; got {a.shape}"
    )


def evaluate_physical_material_points(state, ops, *, K0, GB, quadrature):
    """
    Reconstruct epsilon_i, sigma_i and detF at all thickness Gauss points
    from a converged TE1 state and the SAME frozen GB field.

    J-16 audit optimization: keep only the thickness-quadrature loop in
    Python; all midsurface nodes of one Gauss plane are evaluated by NumPy in
    one batch.  The TE1 kinematics and Birger constitutive equations are
    unchanged.
    """
    q,qx,qs=c3.profile_fields_and_first_derivatives(state,ops)
    q=np.asarray(q,float); qx=np.asarray(qx,float); qs=np.asarray(qs,float)
    surf_shape=(ops.grid.ns,ops.grid.nx)
    expected=(6,)+surf_shape
    if q.shape != expected or qx.shape != expected or qs.shape != expected:
        raise ValueError(f"TE1 reconstructed fields must have shape {expected}")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(qx)) and np.all(np.isfinite(qs))):
        raise ValueError("non-finite reconstructed TE1 fields")

    R=float(ops.grid.radius); K0=float(K0)
    if R <= 0.0 or K0 <= 0.0:
        raise ValueError("R and K0 must be positive")

    nz=quadrature.z.size
    GBa=np.asarray(GB,float)
    if GBa.ndim == 0:
        GB_mode="scalar"
    elif GBa.shape == surf_shape:
        GB_mode="surface"
    elif GBa.shape == (nz,)+surf_shape:
        GB_mode="full"
    else:
        raise ValueError(
            f"GB must be scalar, surface-shaped {surf_shape}, or {(nz,)+surf_shape}; got {GBa.shape}"
        )
    if np.any(GBa <= 0.0) or not np.all(np.isfinite(GBa)):
        raise ValueError("GB must be finite and positive")

    eps=np.empty((nz,)+surf_shape,float)
    sig=np.empty_like(eps)
    det=np.empty_like(eps)
    I3=np.eye(3,dtype=float)

    # Component views; each is one complete midsurface field.
    _u0,u1,v0,v1,w0,w1=q
    u0x,u1x,v0x,v1x,w0x,w1x=qx
    u0s,u1s,v0s,v1s,w0s,w1s=qs

    for g,z_raw in enumerate(quadrature.z):
        z=float(z_raw)
        H=1.0+z/R
        if H <= 0.0:
            raise ValueError("H=1+z/R must be positive")

        v=v0+z*v1
        w=w0+z*w1
        ux=u0x+z*u1x
        vx=v0x+z*v1x
        wx=w0x+z*w1x
        us=u0s+z*u1s
        vs=v0s+z*v1s
        ws=w0s+z*w1s

        F=np.empty(surf_shape+(3,3),dtype=float)
        F[...,0,0]=1.0+ux
        F[...,0,1]=us/H
        F[...,0,2]=u1
        F[...,1,0]=vx
        F[...,1,1]=1.0+(vs+w/R)/H
        F[...,1,2]=v1
        F[...,2,0]=wx
        F[...,2,1]=(ws-v/R)/H
        F[...,2,2]=1.0+w1

        detg=np.linalg.det(F)
        bad=np.argwhere((~np.isfinite(detg)) | (detg <= 0.0))
        if bad.size:
            is_,ix=map(int,bad[0])
            raise s2.InvalidDeformationMapping(
                f"INVALID_DEFORMATION_MAPPING detF={float(detg[is_,ix]):.17g}, "
                f"gauss={g}, s_index={is_}, x_index={ix}"
            )

        E=0.5*(np.swapaxes(F,-1,-2) @ F-I3)
        theta=np.trace(E,axis1=-2,axis2=-1)
        devE=E-(theta[...,None,None]/3.0)*I3
        epsg=np.sqrt(np.maximum((2.0/3.0)*np.sum(devE*devE,axis=(-2,-1)),0.0))

        if GB_mode == "scalar":
            GBg=GBa
        elif GB_mode == "surface":
            GBg=GBa
        else:
            GBg=GBa[g]
        S=K0*theta[...,None,None]*I3+2.0*GBg[...,None,None]*devE if np.ndim(GBg) else K0*theta[...,None,None]*I3+2.0*float(GBg)*devE
        trS=np.trace(S,axis1=-2,axis2=-1)
        devS=S-(trS[...,None,None]/3.0)*I3
        sigg=np.sqrt(np.maximum((3.0/2.0)*np.sum(devS*devS,axis=(-2,-1)),0.0))

        eps[g]=epsg
        sig[g]=sigg
        det[g]=detg

    return MaterialPointState(eps,sig,det)


def solve_physical_birger_load_step(
    initial_state,
    ops,
    *,
    E0,
    nu0,
    quadrature,
    generalized_load,
    bc,
    inner_settings,
    birger_settings,
    material_diagram,
    initial_GB=None,
    previous_accepted_epsilon_i=None,
):
    """
    One physical load step: TE1 Green-Lagrange + Birger + strict internal MVI.
    """
    ec=nk.ElasticConstants(float(E0),float(nu0))
    gb_shape=(quadrature.z.size,ops.grid.ns,ops.grid.nx)
    if initial_GB is None:
        GB0=np.full(gb_shape,ec.G0,float)
    else:
        a=np.asarray(initial_GB,float)
        if a.ndim == 0:
            GB0=np.full(gb_shape,float(a),float)
        else:
            GB0=np.broadcast_to(a,gb_shape).astype(float,copy=True)

    p=np.asarray(generalized_load,float)
    expected=(6,ops.grid.ns,ops.grid.nx)
    if p.shape != expected:
        raise ValueError(f"generalized_load must have shape {expected}")

    def inner(st,GBfield):
        residual_fn=c4.make_physical_residual_function(
            ops,
            K0=ec.K0,
            GB=GBfield,
            quadrature=quadrature,
            generalized_load=p,
            bc=bc,
        )
        return c4.solve_internal_mvi(
            st,ops.grid,bc,residual_fn,inner_settings,
            quadrature=quadrature
        )

    def evaluator(st,GBfield):
        return evaluate_physical_material_points(
            st,ops,K0=ec.K0,GB=GBfield,quadrature=quadrature
        )

    return solve_birger_outer_cycle(
        initial_state,
        GB0,
        G0=ec.G0,
        material_diagram=material_diagram,
        inner_solve=inner,
        evaluate_material=evaluator,
        state_vector=lambda st: c3.mvi.reconstruct_vector(st,ops.grid),
        settings=birger_settings,
        previous_accepted_epsilon_i=previous_accepted_epsilon_i,
    )
