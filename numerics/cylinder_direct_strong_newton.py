"""Independent experimental Newton driver for direct strong cylindrical FDM.

A distinct strong-form discretization (not weighted-adjoint variational
FDM) supplies all field equations, including explicit axial boundary rows.
The caller must certify axial principal orders and supply ALL physical end
conditions. No automatic order classification, gauge, artificial penalty,
pseudoinverse, default mechanical support, or silent result acceptance.

This research-only driver is deliberately NOT dispatched by FINAL_CASE/GUI.
No claim of continuum well-posedness, model accuracy, mesh convergence or
numerical certification follows from a successful algebraic solve.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


@dataclass(frozen=True)
class StrongNewtonSettings:
    max_dofs: int = 512
    max_iterations: int = 15
    difference_step: float = 1e-6
    relative_linear_tolerance: float = 1e-8
    scaled_absolute_tolerance: float = 1e-8
    line_search_halvings: int = 15
    linear_solver: str = "AUTO"
    dense_lu_dof_limit: int = 384

    def __post_init__(self):
        if self.linear_solver not in ("AUTO","MATRIX_FREE_GMRES","DENSE_LU"):
            raise ValueError("DIRECT_STRONG_UNKNOWN_LINEAR_SOLVER")
        if (type(self.dense_lu_dof_limit) is not int or
                not 1<=self.dense_lu_dof_limit<=2048):
            raise ValueError("DIRECT_STRONG_DENSE_LU_LIMIT_INVALID")
        for key in ('max_dofs', 'max_iterations', 'line_search_halvings'):
            val = getattr(self, key)
            if type(val) is not int or val < (0 if key == 'line_search_halvings' else 1):
                raise ValueError(f'{key}: positive integer required (zero allowed for halvings)')
        for key in ('difference_step', 'relative_linear_tolerance',
                    'scaled_absolute_tolerance'):
            val = getattr(self, key)
            if isinstance(val, (bool, np.bool_)) or not math.isfinite(float(val)) or float(val) <= 0:
                raise ValueError(f'{key}: positive finite value required')
        if self.relative_linear_tolerance >= 1 or self.difference_step >= .1:
            raise ValueError('finite-difference/linear tolerances must be below 1')


@dataclass(frozen=True)
class StrongNewtonResult:
    mechanical: np.ndarray
    electrical: np.ndarray | None
    iterations: int
    converged: bool
    status: str
    scaled_equation_norm: float
    spatial_method_status: str = 'EXPERIMENTAL_DIRECT_STRONG_NEWTON_NOT_CERTIFIED'
    gauge_multiplier_scaled: float | None = None
    gauge_defect_scaled: float | None = None


@dataclass(frozen=True)
class StrongElectricGauge:
    """One volume-mean potential constraint, with explicit CUF quadrature.

    weights has the same (mode,nx,ns) shape as the potential coefficients.
    It represents integral(H*G_k*dA*dz)/volume; higher modes must be kept.
    potential_scale_v scales the single constraint; target_v is its mean.
    """
    weights: np.ndarray
    target_v: float = 0.
    potential_scale_v: float = 1.

    def __post_init__(self):
        w=np.asarray(self.weights,float)
        if w.ndim!=3 or min(w.shape)<1 or not np.isfinite(w).all():
            raise ValueError('finite electric gauge weights (mode,nx,ns) required')
        if not np.isclose(np.sum(w[0]),1.,rtol=1.e-10,atol=1.e-12):
            raise ValueError('constant-mode gauge weights must sum to one')
        if not np.isfinite(self.target_v) or not np.isfinite(self.potential_scale_v) or self.potential_scale_v<=0:
            raise ValueError('finite gauge target and positive voltage scale required')
        w=w.copy();w.setflags(write=False)
        object.__setattr__(self,'weights',w)


def cuf_mean_potential_gauge(surface_weights, *, radius_m, thickness_m,
                             potential_order, target_v=0., potential_scale_v=1.):
    """Exact H-weighted CUF mean for G_k=(2z/h)^k (algorithm §§24,33).

    Odd thickness modes contribute h/(2R*(k+2)); constraining only psi_0
    would therefore impose a different gauge on a curved shell.
    """
    area=np.asarray(surface_weights,float)
    if area.ndim!=2 or min(area.shape)<1 or not np.isfinite(area).all() or np.any(area<=0):
        raise ValueError('positive finite surface quadrature weights required')
    if (type(potential_order) is not int or not 0<=potential_order<=4 or
            not np.isfinite(radius_m) or not np.isfinite(thickness_m) or
            not radius_m>thickness_m/2>0):
        raise ValueError('require potential order 0..4 and R>h/2>0')
    moments=np.array([1./(k+1) if k%2==0 else thickness_m/(2*radius_m*(k+2))
                      for k in range(potential_order+1)])
    return StrongElectricGauge(moments[:,None,None]*(area/np.sum(area)),
                               target_v,potential_scale_v)


def _positive_scale(value, shape, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ': finite positive scale required')
    scale = np.asarray(value, dtype=float)
    if scale.ndim == 0:
        scale = np.full(shape, float(scale))
    if scale.shape != shape or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(name + ': finite positive scale with state shape required')
    return scale


def solve_strong_nonlinear_equations(
        equation: Callable, mechanical, electrical=None, *,
        mechanical_state_scale, mechanical_equation_scale,
        electrical_state_scale=None, electrical_equation_scale=None,
        electric_gauge=None,
        settings=StrongNewtonSettings()):
    """Solve an explicit coupled mechanical/electric strong algebraic system.

    equation(q,p) -> (mechanical equations, electrical equations or None)
    is evaluated with NO spatial-method substitution. Scaling is supplied
    explicitly in both physical units; the operator cannot guess appropriate
    dimensional scales for electromechanics.
    """
    if not isinstance(settings, StrongNewtonSettings):
        raise TypeError('StrongNewtonSettings required')
    q0 = np.asarray(mechanical, dtype=float)
    if q0.ndim != 3 or min(q0.shape) < 1 or not np.isfinite(q0).all():
        raise ValueError('finite mechanical (component,nx,ns) state required')
    p0 = None if electrical is None else np.asarray(electrical, dtype=float)
    if p0 is not None and (p0.ndim != 3 or p0.shape[1:] != q0.shape[1:] or
                                not np.isfinite(p0).all()):
        raise ValueError('finite electrical (mode,nx,ns) state required')
    if electric_gauge is not None and (
            not isinstance(electric_gauge,StrongElectricGauge) or p0 is None or
            electric_gauge.weights.shape != p0.shape):
        raise ValueError('electric gauge must match active potential field')
    if (p0 is None) != (electrical_state_scale is None) or (
            p0 is None) != (electrical_equation_scale is None):
        raise ValueError('electrical state and both dimensional scales must agree')
    qscale = _positive_scale(mechanical_state_scale, q0.shape, 'mechanical state')
    qforce = _positive_scale(mechanical_equation_scale, q0.shape, 'mechanical equation')
    if p0 is None:
        xscale = qscale.ravel()
        rscale = qforce.ravel()
    else:
        pscale = _positive_scale(electrical_state_scale, p0.shape, 'electric state')
        pforce = _positive_scale(electrical_equation_scale, p0.shape, 'electric equation')
        xscale = np.concatenate((qscale.ravel(), pscale.ravel()))
        rscale = np.concatenate((qforce.ravel(), pforce.ravel()))
    state_count = len(xscale)
    count = state_count + int(electric_gauge is not None)
    if count > settings.max_dofs:
        raise ValueError('DIRECT_STRONG_DOF_LIMIT: set explicit affordable max_dofs')
    shape_q = q0.shape
    shape_p = None if p0 is None else p0.shape

    def decode(z):
        flat = z[:state_count] * xscale
        q = flat[:q0.size].reshape(shape_q)
        p = None if shape_p is None else flat[q0.size:].reshape(shape_p)
        return q, p

    gauge_column=None
    if electric_gauge is not None:
        gauge_column=np.concatenate((np.zeros(q0.size),
            electric_gauge.weights.ravel()*pscale.ravel()/electric_gauge.potential_scale_v))

    def evaluate(z, *, original=False):
        if z.shape != (count,) or not np.isfinite(z).all():
            raise ValueError('nonfinite scaled Newton state')
        q, p = decode(z)
        outputs = equation(q, p)
        if (not isinstance(outputs, tuple) or len(outputs) != 2):
            raise TypeError('strong equation must return mechanical/electric tuple')
        rm = np.asarray(outputs[0], float)
        if rm.shape != shape_q or not np.isfinite(rm).all():
            raise ValueError('invalid finite mechanical equations')
        if shape_p is None:
            if outputs[1] is not None:
                raise ValueError('electrical equation when field is OFF')
            full = rm.ravel()
        else:
            re = np.asarray(outputs[1], float)
            if re.shape != shape_p or not np.isfinite(re).all():
                raise ValueError('invalid finite electrical equations')
            full = np.concatenate((rm.ravel(), re.ravel()))
        out = full / rscale
        if not np.isfinite(out).all():
            raise ValueError('nonfinite scaled strong equations')
        if electric_gauge is not None and not original:
            constraint=(float(np.sum(electric_gauge.weights*p))-
                        electric_gauge.target_v)/electric_gauge.potential_scale_v
            out=np.concatenate((out+z[-1]*gauge_column,[constraint]))
        return out

    def finish(z, iterations, converged, status, norm):
        multiplier=defect=None
        if electric_gauge is not None:
            multiplier=float(z[-1])
            _,potential=decode(z)
            defect=abs((float(np.sum(electric_gauge.weights*potential))-
                        electric_gauge.target_v)/electric_gauge.potential_scale_v)
            original_norm=float(np.max(np.abs(evaluate(z,original=True))))
            norm=max(norm,original_norm,defect,abs(multiplier))
            if converged and norm>settings.scaled_absolute_tolerance:
                converged=False
                status='ELECTRIC_GAUGE_COMPATIBILITY_FAILED'
        return StrongNewtonResult(*decode(z),iterations,converged,status,norm,
                                  gauge_multiplier_scaled=multiplier,
                                  gauge_defect_scaled=defect)

    z = np.concatenate((q0.ravel(),p0.ravel()))/xscale if p0 is not None else q0.ravel()/xscale
    if electric_gauge is not None:z=np.concatenate((z,[0.]))
    fun = evaluate(z)
    norm = float(np.max(np.abs(fun)))
    if norm <= settings.scaled_absolute_tolerance:
        return finish(z, 0, True, 'CONVERGED', norm)
    for iteration in range(1, settings.max_iterations + 1):
        # Scaled directional derivative of the SAME physical strong equations.
        # No gradient of a variational weak action is used here.
        def apply(vec):
            magnitude = float(np.linalg.norm(vec))
            if magnitude == 0:
                return np.zeros(count)
            step = settings.difference_step * (1. + float(np.linalg.norm(z))) / magnitude
            # The current deformation can be admissible while a finite
            # difference probe crosses the det(F)>0 boundary. Reduce only
            # the probe, retaining a symmetric derivative at the same state.
            for _ in range(settings.line_search_halvings + 1):
                try:
                    plus = evaluate(z + step*vec)
                    minus = evaluate(z - step*vec)
                    return (plus-minus)/(2*step)
                except ValueError as exc:
                    if "INVALID_DEFORMATION_MAPPING" not in str(exc):
                        raise
                    step *= .5
            raise ValueError('DIRECT_STRONG_INVALID_TANGENT_PROBE')
        try:
            if (settings.linear_solver=="DENSE_LU" or
                    settings.linear_solver=="AUTO" and
                    count<=settings.dense_lu_dof_limit):
                if count>settings.dense_lu_dof_limit:
                    raise ValueError("DIRECT_STRONG_DENSE_LU_DOF_LIMIT")
                # Distinct physical STRONG Jacobian: explicit columns
                # of the same direct-form field equations, not a weak
                # MVI Hessian, approximate inverse or pseudoinverse.
                jacobian=np.column_stack([
                    apply(np.eye(1,count,k).ravel())
                    for k in range(count)])
                if not np.isfinite(jacobian).all():
                    raise ValueError("DIRECT_STRONG_NONFINITE_DENSE_JACOBIAN")
                try:
                    delta=np.linalg.solve(jacobian,-fun)
                    info=0
                except np.linalg.LinAlgError:
                    return finish(z,iteration-1,False,
                                  "SINGULAR_STRONG_TANGENT",norm)
            else:
                jvp=LinearOperator((count,count),matvec=apply,dtype=float)
                delta,info=gmres(
                    jvp,-fun,atol=0.,
                    rtol=settings.relative_linear_tolerance,
                    restart=min(40,count),maxiter=max(10,count//2))
        except ValueError as exc:
            if str(exc)!="DIRECT_STRONG_INVALID_TANGENT_PROBE":
                raise
            return finish(z,iteration-1,False,
                          "INVALID_TANGENT_PERTURBATION",norm)
        if info or not np.isfinite(delta).all():
            return finish(z,iteration-1,False,
                          "LINEAR_TANGENT_NOT_SOLVED",norm)
        accepted = False
        for halves in range(settings.line_search_halvings + 1):
            alpha = 2.**(-halves)
            test = z + alpha*delta
            try:
                check = evaluate(test)
            except (ValueError, FloatingPointError, OverflowError):
                continue
            next_norm = float(np.max(np.abs(check)))
            if next_norm < norm or next_norm <= settings.scaled_absolute_tolerance:
                z,fun,norm = test,check,next_norm
                accepted = True
                break
        if not accepted:
            return finish(z,iteration-1,False,'NEWTON_LINE_SEARCH_FAILED',norm)
        if norm <= settings.scaled_absolute_tolerance:
            return finish(z,iteration,True,'CONVERGED',norm)
    return finish(z,settings.max_iterations,False,'MAX_NEWTON_ITERATIONS',norm)


def solve_direct_strong_reference(functional, mechanical, electrical=None, *,
                                  length_m, axial_orders, mechanical_ends,
                                  electrical_ends=None,
                                  grounded_electric_elsewhere=False,
                                  electric_gauge=None,
                                  mechanical_load=None, electrical_charge=None,
                                  mechanical_state_scale,
                                  mechanical_equation_scale,
                                  electrical_state_scale=None,
                                  electrical_equation_scale=None,
                                  settings=StrongNewtonSettings()):
    """Opt-in nonlinear solver for physical axial strong equations ONLY.

    The supplied per-field differential order and boundary program are an
    EXTERNAL certification obligation. No automatic claim that a given
    KL/FSDT/Reddy/CUF problem has order 2m. Not wired into the GUI.
    """
    from cylinder_direct_strong_fdm import DirectStrongGrid, direct_strong_from_functional
    from cylinder_direct_strong_boundary import assemble_physical_end_rows
    if grounded_electric_elsewhere:
        raise ValueError('Use actual electric end potentials or electric_gauge; a flag does not impose a constraint')
    reference_grid = DirectStrongGrid(mechanical.shape[1], mechanical.shape[2],
                                      length_m, functional.R)

    def equation(q,p):
        balance = direct_strong_from_functional(
            functional,q,p,length_m=length_m,
            mechanical_load=mechanical_load,electrical_charge=electrical_charge)
        rows = assemble_physical_end_rows(
            balance,q,reference_grid,
            axial_orders=axial_orders,mechanical_ends=mechanical_ends,
            electrical=p,electrical_ends=electrical_ends,
            gauge_closed_elsewhere=electric_gauge is not None)
        return rows.mechanical,rows.electrical

    return solve_strong_nonlinear_equations(
        equation,mechanical,electrical,
        mechanical_state_scale=mechanical_state_scale,
        mechanical_equation_scale=mechanical_equation_scale,
        electrical_state_scale=electrical_state_scale,
        electrical_equation_scale=electrical_equation_scale,
        electric_gauge=electric_gauge,
        settings=settings)

__all__ = ('StrongNewtonSettings','StrongNewtonResult','StrongElectricGauge',
           'cuf_mean_potential_gauge',
           'solve_strong_nonlinear_equations','solve_direct_strong_reference')
