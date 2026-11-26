"""Gao--Park (2007): isolated native FDM/MVI benchmark, research only.
Exact plane-strain symmetry restriction u_x=u_s=0, radial CUF coefficients
with free axial traces. No change to working CASE dispatch or solver files.
FDM uses the native matched strong bulk and SAT, extending the zero-order
natural trace lift to radial fields. MVI uses native fixed-rank Y/X sweeps.
"""
import argparse
import json
from pathlib import Path
import numpy as np

A, B, E, NU, P = 1e-6, 5e-6, 135e9, .3, 10e6
R, H, L = (A+B)/2, B-A, 12e-6
ELL = (0., .1e-6, np.sqrt(.05)*1e-6, .5e-6)
U0 = P*A/E
NX, NS = 21, 9


def data(order, nx=None, ns=None):
    nx, ns = NX if nx is None else nx, NS if ns is None else ns
    n=order+1
    f=np.zeros((3*n,nx,ns))
    f[2*n:]=P*(A/R)*((-1.)**np.arange(n))[:,None,None]
    return f,tuple(range(2*n,3*n))


def seed(nfield, rank, nx=None, ns=None):
    nx, ns = NX if nx is None else nx, NS if ns is None else ns
    s=2*np.pi*np.arange(ns)/ns
    modes=[np.ones(ns)]
    for k in range(1,5): modes.extend((np.cos(k*s),np.sin(k*s)))
    return np.zeros((nfield,rank,nx)),np.broadcast_to(np.stack(modes[:rank]),(nfield,rank,ns)).copy()


def reference_te_nm(order,ell,xi):
    """Independent one-dimensional polynomial energy, not a native solver."""
    from numpy.polynomial.legendre import Legendre,leggauss
    z,w=leggauss(64)
    rr=3.+2.*z
    w=w*2.*rr
    lam=NU/((1+NU)*(1-2*NU)); g=1/(2*(1+NU))
    polys=[Legendre.basis(k) for k in range(order+1)]
    p=np.array([f(z) for f in polys])
    dp=np.array([.5*f.deriv()(z) for f in polys])
    ddp=np.array([.25*f.deriv(2)(z) for f in polys])
    circ=p/rr; d=dp/rr-p/rr**2
    def gram(v): return (v*w)@v.T
    k=lam*gram(dp+circ)+2*g*(gram(dp)+gram(circ))
    k+=(ell*1e6)**2*(lam*gram(ddp+d)+2*g*(gram(ddp)+3*gram(d)))
    c=np.linalg.solve(k,np.array([f(-1.) for f in polys]))
    return sum(a*f(xi) for a,f in zip(c,polys))*U0*1e9


def solve_fdm(order, ell):
    from cylinder_direct_strong_model import DirectStrongShell
    from cylinder_direct_strong_fdm import DirectStrongGrid
    from cylinder_direct_strong_newton import StrongNewtonSettings, solve_strong_nonlinear_equations
    from cylinder_matched_strong_sat import add_natural_trace_sat
    from flexo_gradient_g3d import G3DMaterial
    g=DirectStrongGrid(NX,NS,L,R,axial_scheme='MATCHED_QUARTIC_SAT_V1')
    model=DirectStrongShell(f'CUF_TE{order}',G3DMaterial(E,NU,ell,0.,0.,0.,'G3D'),grid=g,thickness_m=H,n_gauss=32,geometry_model='LINEAR')
    load,active=data(order); fixed=2*(order+1)
    scale=np.full(model.shape,P); scale[:fixed]=U0
    t0=np.eye(NX)[[0,-1]]
    t1=np.array([-1.,1.])[:,None]*g.matched_axial.kinematic[1][[0,-1]]/g.dx
    def equation(q,phi):
        b=model.balance(q,mechanical_load=load)
        a0=b.mechanical_end_displacement.copy(); a0[:,:fixed]=0.
        a1=b.mechanical_end_slope.copy(); a1[:,:fixed]=0.
        residual=add_natural_trace_sat(b.mechanical_bulk,g.axial_weights,(t0,t1),(a0,a1),(np.zeros_like(a0),np.zeros_like(a1)))
        residual[:fixed]=q[:fixed]
        return residual,None
    settings=StrongNewtonSettings(max_dofs=5000,max_iterations=12,axisymmetric=True,linear_solver='MATRIX_FREE_GMRES',scaled_absolute_tolerance=1e-8)
    ans=solve_strong_nonlinear_equations(equation,np.zeros(model.shape),None,mechanical_state_scale=U0,mechanical_equation_scale=scale,settings=settings)
    return ans.mechanical,ans.converged,ans.status


def solve_mvi(order,ell,rank):
    from flexo_gradient_g3d import G3DMaterial
    from independent_mvi.coupled_surface import CoupledMVISurfaceGrid,NativeCoupledMVISurface,CoupledMVIFactors
    from independent_mvi.coupled_halfstep import CoupledMVIEndProgram,CoupledMVIHalfstepSettings
    from independent_mvi.coupled_static import CoupledMVISweepSettings,solve_coupled_mvi_static
    load,active=data(order); fixed=2*(order+1)
    class PlaneStrainEnds(CoupledMVIEndProgram):
        def validate(self,surface):
            if self.mechanical!='GAO_PARK_RADIAL_FREE' or self.electric!='OFF' or surface.electrical_on or surface.theory!=f'CUF_TE{order}':
                raise ValueError('GAO_PARK_BOUNDARY_MISMATCH')
        def free_axial_indices(self,surface):
            self.validate(surface)
            return tuple(np.arange(1,NX-1) if c<fixed else np.arange(NX) for c in range(surface.n_fields))
    surface=NativeCoupledMVISurface(f'CUF_TE{order}',G3DMaterial(E,NU,ell,0.,0.,0.,'G3D'),grid=CoupledMVISurfaceGrid(NX,NS,L,R),thickness_m=H,n_gauss=32,geometry_model='LINEAR',potential_order=None)
    x,y=seed(surface.n_fields,rank)
    factors=CoupledMVIFactors(x,y)
    settings=CoupledMVISweepSettings(max_sweeps=8,relative_field_change=1e-7,halfstep=CoupledMVIHalfstepSettings(max_iterations=12,max_coordinates=2000,linear_solver='MATRIX_FREE_GMRES',tolerance_scaled=1e-8,krylov_restart=48,krylov_maxiter=80,krylov_relative_tolerance=1e-7),allow_projected_convergence=False)
    ans=solve_coupled_mvi_static(surface,factors,program=PlaneStrainEnds('GAO_PARK_RADIAL_FREE','OFF'),field_scales=[U0]*surface.n_fields,energy_scale_j=P*U0*2*np.pi*R*L,mechanical_load=load,settings=settings,active_fields=active)
    if ans.factors.rank!=rank: raise RuntimeError('GAO_PARK_RANK_CHANGED')
    return surface.reconstruct(ans.factors),ans.converged,ans.status


def run(order,ell_index,method,rank=None):
    ell=ELL[ell_index]
    if method=='FDM': q,ok,status=solve_fdm(order,ell)
    else: q,ok,status=solve_mvi(order,ell,rank)
    n=order+1; c=q[2*n:]; xi=np.linspace(-1.,1.,401)
    profile=np.einsum('zk,kij->zij',xi[:,None]**np.arange(n),c)*1e9
    mean=profile.mean(axis=(1,2))
    reference=reference_te_nm(order,ell,xi)
    max_error=float(np.max(np.abs(profile-reference[:,None,None]))/np.max(np.abs(reference)))
    deviation=float(np.max(np.abs(profile-mean[:,None,None])))
    result=dict(order=order,ell_um=float(ell*1e6),method=method,rank=rank,nx=NX,ns=NS,grid=f'{NX}x{NS}',geometry='LINEAR',gradient_model='G3D',n_gauss=32,converged=bool(ok),status=status,inner_nm=float(mean[0]),mid_nm=float(mean[200]),outer_nm=float(mean[-1]),max_axial_angular_variation_nm=deviation,reference_te_inner_nm=float(reference[0]),max_profile_relative_error=max_error,benchmark_pass=bool(ok and max_error<1e-3))
    return result,profile


def main():
    global NX, NS
    p=argparse.ArgumentParser()
    p.add_argument('--order',type=int,choices=range(1,5),required=True)
    p.add_argument('--ell-index',type=int,choices=range(4),required=True)
    p.add_argument('--method',choices=['FDM','MVI'],required=True)
    p.add_argument('--rank',type=int,choices=range(1,9))
    p.add_argument('--nx',type=int,default=21)
    p.add_argument('--ns',type=int,default=9)
    p.add_argument('--output',default='gao_park_native')
    a=p.parse_args()
    if not 20 <= a.nx <= 257 or a.ns < 9 or a.ns % 2 == 0:
        p.error('matched FDM needs 20<=nx<=257; fixed ranks 1-8 require odd ns>=9')
    NX, NS = a.nx, a.ns
    if (a.method=='MVI')!=(a.rank is not None): p.error('rank is required only for MVI')
    from cylinder_paths import activate
    activate()
    result,profile=run(a.order,a.ell_index,a.method,a.rank)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'profile.npz',profile_nm=profile,radius_um=(R+H/2*np.linspace(-1.,1.,401))*1e6, nx=NX, ns=NS)
    print(json.dumps(result),flush=True)
    if not result['benchmark_pass']: raise SystemExit(2)


if __name__=='__main__': main()
