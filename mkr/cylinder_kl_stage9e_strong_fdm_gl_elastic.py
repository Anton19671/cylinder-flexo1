#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import final_progress as fprog

import final_numeric_scaling as nscale
import final_v3_fd as v3fd
import final_v3_analytic as v3an
import final_residual_scaling as rscale
import final_residual_regions as rregions

import cylinder_kl_stage8a_elastic_mechanics as kl
import cylinder_kl_full_gl_kinematics as fullgl
import cylinder_kl_stage8b1_boundary as kb
import cylinder_kl_stage9b_strong_fdm_linear as fl


class KLStrongFDMNonlinearError(RuntimeError):
    pass


class KLStrongFDMNonlinearSystemSingular(KLStrongFDMNonlinearError):
    pass


@dataclass(frozen=True)
class KLStrongFDMNewtonSettings:
    residual_tolerance_relative: float
    increment_tolerance: float
    fd_step: float
    max_iterations: int

    def __post_init__(self):
        if self.residual_tolerance_relative <= 0.0:
            raise ValueError("residual_tolerance_relative must be >0")
        if self.increment_tolerance <= 0.0:
            raise ValueError("increment_tolerance must be >0")
        if self.fd_step <= 0.0:
            raise ValueError("fd_step must be >0")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >=1")


@dataclass
class KLStrongFDMNonlinearResult:
    U: np.ndarray
    converged: bool
    status: str
    iterations: int
    residual_inf: float
    residual_relative: float
    residual_scale: float
    increment_inf: float
    history: list
    residual_inf_interior: float = 0.0
    residual_inf_boundary: float = 0.0
    residual_inf_load_front: float = 0.0
    residual_convergence_inf: float = 0.0
    residual_convergence_relative: float = 0.0


def strong_collocation_residual_gl(
    U, ops, bc, *, E, nu, h, generalized_load,
    resultants_method="ANALYTIC", quadrature=None
):
    fl._validate_grid(ops)
    U = fl._validate_U(U, ops)
    bc.check_rigid_mode_requirement()

    p = np.asarray(generalized_load, float)
    expected = (3, ops.grid.ns, ops.grid.nx)
    if p.shape != expected:
        raise ValueError(f"generalized_load must have shape {expected}")
    if not np.all(np.isfinite(p)):
        raise ValueError("generalized_load contains non-finite values")

    kin = kl.field_kinematics(U, ops, geometry="GREEN_LAGRANGE")
    const = kl.KLElasticConstants(E, nu, h)
    method = str(resultants_method).upper()
    if method == "ANALYTIC":
        nm = kl.analytic_elastic_resultants(kin.state, const)
    elif method == "QUADRATURE":
        if quadrature is None:
            raise ValueError("quadrature required")
        nm = kl.quadrature_elastic_resultants(kin.state, const, quadrature)
    else:
        raise ValueError("bad resultants_method")

    if kin.full_gl is None:
        raise KLStrongFDMNonlinearError("full Green-Lagrange KL kinematics is missing")
    strong, _full_gl_coefficients = fullgl.variational_strong_residual(
        kin.full_gl, nm, ops, p
    )

    ql = kb.end_mechanical_quantities(
        kin, nm, ops, geometry="GREEN_LAGRANGE", side="LEFT"
    )
    qr = kb.end_mechanical_quantities(
        kin, nm, ops, geometry="GREEN_LAGRANGE", side="RIGHT"
    )

    ns, nx = ops.grid.ns, ops.grid.nx

    lu = fl._bc_component_residual(bc.left.u, ql.u, ql.T_u, ns)
    ru = fl._bc_component_residual(bc.right.u, qr.u, qr.T_u, ns)
    lv = fl._bc_component_residual(bc.left.v, ql.v, ql.T_v, ns)
    rv = fl._bc_component_residual(bc.right.v, qr.v, qr.T_v, ns)
    lw = fl._bc_component_residual(bc.left.w, ql.w, ql.T_w, ns)
    rw = fl._bc_component_residual(bc.right.w, qr.w, qr.T_w, ns)
    ls = fl._bc_component_residual(
        bc.left.slope, ql.theta_n, ql.T_theta, ns
    )
    rs = fl._bc_component_residual(
        bc.right.slope, qr.theta_n, qr.T_theta, ns
    )

    rows = []
    for js in range(ns):
        rows.append(lu[js])
        rows.extend(strong[0, js, 1:nx-1])
        rows.append(ru[js])

    for js in range(ns):
        rows.append(lv[js])
        rows.extend(strong[1, js, 1:nx-1])
        rows.append(rv[js])

    for js in range(ns):
        rows.append(lw[js])
        rows.append(ls[js])
        rows.extend(strong[2, js, 2:nx-2])
        rows.append(rs[js])
        rows.append(rw[js])

    R = np.asarray(rows, float)
    expected_n = 3*ns*nx
    if R.shape != (expected_n,):
        raise KLStrongFDMNonlinearError("residual size mismatch")
    return R, strong, nm, kin


def finite_difference_jacobian(
    U, ops, bc, *, E, nu, h, generalized_load, fd_step,
    resultants_method="ANALYTIC", quadrature=None, base_residual=None
):
    """V3 scaled central FD Jacobian; fd_step is dimensionless/relative."""
    U = fl._validate_U(U, ops)
    x = fl.flatten_U(U, ops)
    if base_residual is None:
        r0, *_ = strong_collocation_residual_gl(
            U, ops, bc, E=E, nu=nu, h=h,
            generalized_load=generalized_load,
            resultants_method=resultants_method, quadrature=quadrature,
        )
    else:
        r0 = np.asarray(base_residual, float).reshape(-1)
        if r0.shape != (x.size,):
            raise ValueError("base_residual size mismatch")

    scales=v3fd.kl_characteristic_scale(U,h).reshape(-1)
    steps=v3fd.relative_steps(x,scales,fd_step)
    J = np.empty((x.size, x.size), float)
    for j,hj in enumerate(steps):
        xp=x.copy(); xm=x.copy()
        xp[j]+=hj; xm[j]-=hj
        rp, *_ = strong_collocation_residual_gl(
            fl.unflatten_U(xp,ops), ops, bc, E=E, nu=nu, h=h,
            generalized_load=generalized_load,
            resultants_method=resultants_method, quadrature=quadrature,
        )
        rm, *_ = strong_collocation_residual_gl(
            fl.unflatten_U(xm,ops), ops, bc, E=E, nu=nu, h=h,
            generalized_load=generalized_load,
            resultants_method=resultants_method, quadrature=quadrature,
        )
        J[:, j] = (rp-rm)/(2.0*hj)
    return J, r0



def _exact_tangent_kinematics(dU,ops,base_kin):
    """Exact directional derivative of full KL Green-Lagrange kinematics."""
    dU=fl._validate_U(dU,ops)
    du,dv,dw=dU
    dux=ops.Dx(du); dus=ops.Ds(du)
    dvx=ops.Dx(dv); dvs=ops.Ds(dv)
    dwx=ops.Dx(dw); dws=ops.Ds(dw)
    dwxx=ops.Dxx(dw); dwss=ops.Dss(dw); dwxs=ops.Ds(dwx)
    R=float(ops.grid.radius)
    v0=base_kin.U[1]; w0=base_kin.U[2]
    a=1.0+base_kin.vs+w0/R
    b=base_kin.ws-v0/R
    da=dvs+dw/R
    db=dws-dv/R
    ds=kl.KLMidSurfaceState(
        eps_x=(1.0+base_kin.ux)*dux + base_kin.vx*dvx + base_kin.wx*dwx,
        eps_s=a*da + base_kin.us*dus + b*db,
        gamma_xs=(dux*base_kin.us + (1.0+base_kin.ux)*dus
                  + dvx*a + base_kin.vx*da
                  + dwx*b + base_kin.wx*db),
        kappa_x=-dwxx,
        kappa_s=-dwss,
        kappa_xs=-dwxs,
    )
    return kl.KLFieldKinematics(
        dU,dux,dus,dvx,dvs,dwx,dws,dwxx,dwss,dwxs,ds
    )


def _exact_tangent_resultants(dkin,const,*,resultants_method,quadrature):
    method=str(resultants_method).upper()
    if method=="ANALYTIC":
        return kl.analytic_elastic_resultants(dkin.state,const)
    if method=="QUADRATURE":
        if quadrature is None:
            raise ValueError("quadrature required")
        return kl.quadrature_elastic_resultants(dkin.state,const,quadrature)
    raise ValueError("bad resultants_method")


def _exact_tangent_strong(base_kin,base_nm,dkin,dnm,ops):
    """Exact directional derivative of full GL strong equilibrium residual."""
    R=float(ops.grid.radius)
    v0=base_kin.U[1]; w0=base_kin.U[2]
    dv=dkin.U[1]; dw=dkin.U[2]
    a=1.0+base_kin.vs+w0/R
    b=base_kin.ws-v0/R
    da=dkin.vs+dw/R
    db=dkin.ws-dv/R

    dfx_u=(dnm.Nx*(1.0+base_kin.ux)+base_nm.Nx*dkin.ux
           +dnm.Nxs*base_kin.us+base_nm.Nxs*dkin.us)
    dfs_u=(dnm.Ns*base_kin.us+base_nm.Ns*dkin.us
           +dnm.Nxs*(1.0+base_kin.ux)+base_nm.Nxs*dkin.ux)
    dRu=-ops.Dx(dfx_u)-ops.Ds(dfs_u)

    dfx_v=(dnm.Nx*base_kin.vx+base_nm.Nx*dkin.vx
           +dnm.Nxs*a+base_nm.Nxs*da)
    dfs_v=(dnm.Ns*a+base_nm.Ns*da
           +dnm.Nxs*base_kin.vx+base_nm.Nxs*dkin.vx)
    dzerov=(dnm.Ns*b+base_nm.Ns*db
            +dnm.Nxs*base_kin.wx+base_nm.Nxs*dkin.wx)/R
    dRv=-ops.Dx(dfx_v)-ops.Ds(dfs_v)-dzerov

    dfx_w=(dnm.Nx*base_kin.wx+base_nm.Nx*dkin.wx
           +dnm.Nxs*b+base_nm.Nxs*db)
    dfs_w=(dnm.Ns*b+base_nm.Ns*db
           +dnm.Nxs*base_kin.wx+base_nm.Nxs*dkin.wx)
    dmem=-ops.Dx(dfx_w)-ops.Ds(dfs_w)
    dzerow=(dnm.Ns*a+base_nm.Ns*da
            +dnm.Nxs*base_kin.vx+base_nm.Nxs*dkin.vx)/R
    dbend=(-ops.Dxx(dnm.Mx)
           -2.0*ops.Ds(ops.Dx(dnm.Mxs))
           -ops.Dss(dnm.Ms))
    dRw=dmem+dzerow+dbend
    return np.stack([dRu,dRv,dRw],axis=0)


def _bc_tangent_value(cond,dgeom,dnatural):
    if cond.kind=="GEOMETRIC":
        return np.asarray(dgeom,float)
    if cond.kind=="NATURAL":
        return np.asarray(dnatural,float)
    if cond.kind=="ELASTIC":
        ns=np.asarray(dgeom,float).size
        return np.asarray(dnatural,float)+cond.stiffness_array(ns)*np.asarray(dgeom,float)
    raise ValueError(f"unsupported KL boundary kind: {cond.kind}")


def _exact_tangent_end(base_kin,base_nm,dkin,dnm,ops,*,side):
    ix,n=kb._edge_index(side)
    R=float(ops.grid.radius)
    v0=base_kin.U[1]; w0=base_kin.U[2]
    dv=dkin.U[1]; dw=dkin.U[2]
    a=1.0+base_kin.vs+w0/R
    b=base_kin.ws-v0/R
    da=dkin.vs+dw/R
    db=dkin.ws-dv/R
    dflux_u=(dnm.Nx*(1.0+base_kin.ux)+base_nm.Nx*dkin.ux
             +dnm.Nxs*base_kin.us+base_nm.Nxs*dkin.us)
    dflux_v=(dnm.Nx*base_kin.vx+base_nm.Nx*dkin.vx
             +dnm.Nxs*a+base_nm.Nxs*da)
    dAx=(dnm.Nx*base_kin.wx+base_nm.Nx*dkin.wx
         +dnm.Nxs*b+base_nm.Nxs*db)
    return dict(
        T_u=np.asarray(n*dflux_u[:,ix],float),
        T_v=np.asarray(n*dflux_v[:,ix],float),
        T_w=np.asarray(n*(dAx[:,ix]+ops.Dx(dnm.Mx)[:,ix]+2.0*ops.Ds(dnm.Mxs)[:,ix]),float),
        T_theta=np.asarray(-dnm.Mx[:,ix],float),
        u=np.asarray(dkin.U[0,:,ix],float),
        v=np.asarray(dkin.U[1,:,ix],float),
        w=np.asarray(dkin.U[2,:,ix],float),
        theta_n=np.asarray(n*dkin.wx[:,ix],float),
    )


def _prepare_exact_elastic_tangent_action(
    U,ops,bc,*,E,nu,h,resultants_method="ANALYTIC",quadrature=None
):
    U=fl._validate_U(U,ops)
    base_kin=kl.field_kinematics(U,ops,geometry="GREEN_LAGRANGE")
    const=kl.KLElasticConstants(E,nu,h)
    method=str(resultants_method).upper()
    if method=="ANALYTIC":
        base_nm=kl.analytic_elastic_resultants(base_kin.state,const)
    elif method=="QUADRATURE":
        if quadrature is None:
            raise ValueError("quadrature required")
        base_nm=kl.quadrature_elastic_resultants(base_kin.state,const,quadrature)
    else:
        raise ValueError("bad resultants_method")
    ns,nx=ops.grid.ns,ops.grid.nx

    def action(dU):
        dkin=_exact_tangent_kinematics(dU,ops,base_kin)
        dnm=_exact_tangent_resultants(
            dkin,const,resultants_method=method,quadrature=quadrature
        )
        dstrong=_exact_tangent_strong(base_kin,base_nm,dkin,dnm,ops)
        ql=_exact_tangent_end(base_kin,base_nm,dkin,dnm,ops,side="LEFT")
        qr=_exact_tangent_end(base_kin,base_nm,dkin,dnm,ops,side="RIGHT")
        lu=_bc_tangent_value(bc.left.u,ql['u'],ql['T_u'])
        ru=_bc_tangent_value(bc.right.u,qr['u'],qr['T_u'])
        lv=_bc_tangent_value(bc.left.v,ql['v'],ql['T_v'])
        rv=_bc_tangent_value(bc.right.v,qr['v'],qr['T_v'])
        lw=_bc_tangent_value(bc.left.w,ql['w'],ql['T_w'])
        rw=_bc_tangent_value(bc.right.w,qr['w'],qr['T_w'])
        ls=_bc_tangent_value(bc.left.slope,ql['theta_n'],ql['T_theta'])
        rs=_bc_tangent_value(bc.right.slope,qr['theta_n'],qr['T_theta'])
        rows=[]
        for js in range(ns):
            rows.append(lu[js]); rows.extend(dstrong[0,js,1:nx-1]); rows.append(ru[js])
        for js in range(ns):
            rows.append(lv[js]); rows.extend(dstrong[1,js,1:nx-1]); rows.append(rv[js])
        for js in range(ns):
            rows.append(lw[js]); rows.append(ls[js]); rows.extend(dstrong[2,js,2:nx-2]); rows.append(rs[js]); rows.append(rw[js])
        out=np.asarray(rows,float)
        if out.shape!=(3*ns*nx,):
            raise KLStrongFDMNonlinearError("analytic tangent residual size mismatch")
        return out
    return action


def analytic_tangent_action(
    U,dU,ops,bc,*,E,nu,h,generalized_load=None,
    resultants_method="ANALYTIC",quadrature=None
):
    """Unsupported: legacy hybrid tangent differs from current GL residual."""
    raise NotImplementedError(
        "GL_ANALYTIC_TANGENT_NOT_CONSISTENT_WITH_CURRENT_RESIDUAL")
    action=_prepare_exact_elastic_tangent_action(
        U,ops,bc,E=E,nu=nu,h=h,resultants_method=resultants_method,
        quadrature=quadrature,
    )
    return action(dU)


def analytic_jacobian(
    U,ops,bc,*,E,nu,h,generalized_load,
    resultants_method="ANALYTIC",quadrature=None,base_residual=None
):
    """Unsupported: do not expose a Jacobian of obsolete GL equations."""
    raise NotImplementedError(
        "GL_ANALYTIC_JACOBIAN_NOT_CONSISTENT_WITH_CURRENT_RESIDUAL")
    U=fl._validate_U(U,ops)
    action=_prepare_exact_elastic_tangent_action(
        U,ops,bc,E=E,nu=nu,h=h,resultants_method=resultants_method,
        quadrature=quadrature,
    )
    J,stats=v3an.assemble_dense_exact_tangent(U,action)
    analytic_jacobian.last_stats=stats.to_dict()
    if base_residual is None:
        r0,*_=strong_collocation_residual_gl(
            U,ops,bc,E=E,nu=nu,h=h,generalized_load=generalized_load,
            resultants_method=resultants_method,quadrature=quadrature,
        )
    else:
        r0=np.asarray(base_residual,float).reshape(-1)
        if r0.shape!=(U.size,):
            raise ValueError("base_residual size mismatch")
    return J,r0


analytic_jacobian.last_stats=None


def solve_gl_elastic_strong_fdm(
    initial_U, ops, bc, *, E, nu, h, generalized_load, settings,
    resultants_method="ANALYTIC", quadrature=None, load_front_surface_mask=None
):
    U = fl._validate_U(initial_U, ops).copy()
    history = []
    last_increment = np.inf

    # V2: one fixed relative-residual scale for the complete Newton solve.
    R_initial, *_ = strong_collocation_residual_gl(
        U, ops, bc, E=E, nu=nu, h=h, generalized_load=generalized_load,
        resultants_method=resultants_method, quadrature=quadrature,
    )
    residual_scale = rscale.residual_scale(generalized_load, R_initial)
    residual_limit = rscale.threshold(settings.residual_tolerance_relative, residual_scale)
    region_masks = rregions.resolve_front_exclusion(rregions.kl_masks(ops, load_front_surface_mask),generalized_load)

    for k in range(1, settings.max_iterations+1):
        # V4: test current residual first; reuse V2 initial residual at k=1.
        if k == 1:
            R = R_initial
        else:
            R, *_ = strong_collocation_residual_gl(
                U, ops, bc, E=E, nu=nu, h=h,
                generalized_load=generalized_load,
                resultants_method=resultants_method, quadrature=quadrature,
            )
        rm = rregions.metrics(R, region_masks, residual_scale)
        rinf = rm.residual_inf

        if rregions.converged(rm, residual_limit):
            return KLStrongFDMNonlinearResult(
                U=U, converged=True,
                status="CONVERGED_STRONG_FDM_GL_ELASTIC",
                iterations=k-1, residual_inf=rinf,
                residual_relative=rm.residual_relative,
                residual_scale=residual_scale,
                increment_inf=0.0, history=history,
                residual_inf_interior=rm.residual_inf_interior,
                residual_inf_boundary=rm.residual_inf_boundary,
                residual_inf_load_front=rm.residual_inf_load_front,
                residual_convergence_inf=rm.residual_convergence_inf,
                residual_convergence_relative=rm.residual_convergence_relative
            )

        # The former analytic tangent differentiated the obsolete
        # exact-membrane/linear-curvature hybrid. Until a second-order
        # consistent tangent of the full normal/curvature map is installed,
        # use the canonical centred derivative of the complete residual.
        J, _ = finite_difference_jacobian(
            U, ops, bc, E=E, nu=nu, h=h,
            generalized_load=generalized_load,fd_step=settings.fd_step,
            resultants_method=resultants_method,
            quadrature=quadrature,base_residual=R,
        )

        try:
            dx, _, _, _ = nscale.equilibrated_solve(J, -R)
        except np.linalg.LinAlgError as exc:
            raise KLStrongFDMNonlinearSystemSingular(
                f"SINGULAR_STRONG_FDM_GL_JACOBIAN / iteration={k}"
            ) from exc

        if not np.all(np.isfinite(dx)):
            raise KLStrongFDMNonlinearError(
                f"NONFINITE_STRONG_FDM_GL_INCREMENT / iteration={k}"
            )

        last_increment = float(np.max(np.abs(dx)))
        U = fl.unflatten_U(fl.flatten_U(U, ops)+dx, ops)

        Rnew, *_ = strong_collocation_residual_gl(
            U, ops, bc, E=E, nu=nu, h=h,
            generalized_load=generalized_load,
            resultants_method=resultants_method,
            quadrature=quadrature,
        )
        rmnew = rregions.metrics(Rnew, region_masks, residual_scale)
        rnew = rmnew.residual_inf

        hrow={"iteration":k,"residual_scale":residual_scale,"increment_inf":last_increment}
        hrow.update(rregions.as_dict(rmnew))
        history.append(hrow)
        _w_live=np.asarray(U[2],float)
        _wis,_wix=np.unravel_index(int(np.argmax(np.abs(_w_live))),_w_live.shape)
        fprog.emit("newton_iteration",
            iteration=int(k), residual_inf=float(hrow.get("residual_inf", rnew)),
            residual_relative=float(hrow.get("residual_relative", 0.0)),
            residual_convergence_inf=float(hrow.get("residual_convergence_inf", hrow.get("residual_inf", rnew))),
            residual_convergence_relative=float(hrow.get("residual_convergence_relative", hrow.get("residual_relative", 0.0))),
            residual_inf_interior=float(hrow.get("residual_inf_interior", 0.0)),
            residual_inf_boundary=float(hrow.get("residual_inf_boundary", 0.0)),
            residual_inf_load_front=float(hrow.get("residual_inf_load_front", 0.0)),
            increment_inf=float(last_increment), stage="NEWTON",
            w_abs_max=float(np.max(np.abs(_w_live))),
            w_at_abs_max=float(_w_live[_wis,_wix]),
            wmax_is=int(_wis),wmax_ix=int(_wix))

        if (
            rregions.converged(rmnew, residual_limit)
            and last_increment <= settings.increment_tolerance
        ):
            return KLStrongFDMNonlinearResult(
                U=U, converged=True,
                status="CONVERGED_STRONG_FDM_GL_ELASTIC",
                iterations=k, residual_inf=rnew,
                residual_relative=rmnew.residual_relative,
                residual_scale=residual_scale,
                increment_inf=last_increment, history=history,
                residual_inf_interior=rmnew.residual_inf_interior,
                residual_inf_boundary=rmnew.residual_inf_boundary,
                residual_inf_load_front=rmnew.residual_inf_load_front,
                residual_convergence_inf=rmnew.residual_convergence_inf,
                residual_convergence_relative=rmnew.residual_convergence_relative
            )

    Rfin, *_ = strong_collocation_residual_gl(
        U, ops, bc, E=E, nu=nu, h=h,
        generalized_load=generalized_load,
        resultants_method=resultants_method,
        quadrature=quadrature,
    )
    rmfin = rregions.metrics(Rfin, region_masks, residual_scale)
    return KLStrongFDMNonlinearResult(
        U=U, converged=False,
        status="MAX_STRONG_FDM_GL_NEWTON_ITERATIONS",
        iterations=settings.max_iterations,
        residual_inf=rmfin.residual_inf,
        residual_relative=rmfin.residual_relative,
        residual_scale=residual_scale,
        increment_inf=float(last_increment),
        history=history,
        residual_inf_interior=rmfin.residual_inf_interior,
        residual_inf_boundary=rmfin.residual_inf_boundary,
        residual_inf_load_front=rmfin.residual_inf_load_front,
        residual_convergence_inf=rmfin.residual_convergence_inf,
        residual_convergence_relative=rmfin.residual_convergence_relative
    )
