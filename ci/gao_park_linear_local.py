"""Optional, independently audited acceleration of the unchanged linear local law.

Only the opt-in Gao-Park benchmark uses it. No production solver edits.
The same CUF constitutive kernel generates every tangent column with the
same Gauss rule, material, geometry and thickness. FDM and MVI keep their
own spatial assembly and solve independently; no field-space substitution.
"""
import json
import numpy as np

FIELDS=('q','qx','qs','qxx','qxs','qss')


class VerifiedLinearCUFLocal:
    def __init__(self,model,u0):
        from cuf_g3d_weak import CUFSurfaceJets,WeakCoefficients
        if (not model.theory.startswith('CUF_TE') or model.geometry_model!='LINEAR'
                or model.electrical_on or model.material.kappa_f_per_m!=0.
                or model.material.mu1_c_per_m!=0. or model.material.mu2_c_per_m!=0.):
            raise ValueError('CACHE_ONLY_LINEAR_MECHANICAL_CUF')
        if not np.isfinite(u0) or u0<=0: raise ValueError('invalid displacement scale')
        self.jet_type=CUFSurfaceJets
        self.work_type=WeakCoefficients
        self.direct=model._local
        self.shape=(3,int(model.theory[-1])+1,6)
        radius=model.grid.radius_m
        self.scale=np.broadcast_to(u0/radius**np.array([0,1,1,2,2,2]),self.shape).copy().ravel()
        self.energy_scale=model.material.young_pa*u0*u0/model.thickness_m
        self.gauss=model.n_gauss
        n=len(self.scale)
        self.K=np.empty((n,n))
        z=np.zeros(n)
        origin=self.direct(self.jets(z))
        if origin.energy_j_per_m2!=0. or np.any(origin.mechanical) or np.any(origin.electrical):
            raise ValueError('CACHE_ORIGIN_NOT_ZERO')
        for i in range(n):
            z[i]=1.
            work=self.direct(self.jets(z))
            self.K[:,i]=work.mechanical.ravel()*self.scale/self.energy_scale
            z[i]=0.
        if not np.isfinite(self.K).all(): raise ValueError('CACHE_TANGENT_NONFINITE')
        symmetry=np.linalg.norm(self.K-self.K.T,ord=np.inf)/np.linalg.norm(self.K,ord=np.inf)
        if symmetry>3e-9: raise ValueError('CACHE_LOCAL_TANGENT_NOT_SYMMETRIC')
        self.K.setflags(write=False)
        self.scale.setflags(write=False)
        rng=np.random.default_rng(20261126)
        worst_work=0.; worst_energy=0.
        # Independent probes include all displacement and derivative fields,
        # not just the radial coefficients or the exact axisymmetric solution.
        for factor in (.1,1.,3.):
            for _ in range(4):
                x=factor*rng.normal(size=n)
                jets=self.jets(x)
                old=self.direct(jets)
                new=self(jets)
                a=old.mechanical.ravel()*self.scale/self.energy_scale
                b=new.mechanical.ravel()*self.scale/self.energy_scale
                err=np.linalg.norm(a-b,np.inf)/max(np.linalg.norm(a,np.inf),1e-30)
                eng=abs(old.energy_j_per_m2-new.energy_j_per_m2)/max(abs(old.energy_j_per_m2),1e-30)
                worst_work=max(worst_work,err);worst_energy=max(worst_energy,eng)
        if worst_work>3e-9 or worst_energy>3e-9:
            raise ValueError(f'CACHE_INDEPENDENT_PROBE_MISMATCH {worst_work} {worst_energy}')
        self.audit=dict(theory=model.theory,gauss=self.gauss,columns=n,
                        relative_symmetry_error=float(symmetry),
                        max_relative_probe_work_error=float(worst_work),
                        max_relative_probe_energy_error=float(worst_energy),probes=12)

    def jets(self,x):
        z=(np.asarray(x)*self.scale).reshape(self.shape)
        return self.jet_type(*(z[:,:,i] for i in range(6)),*(np.zeros(1) for _ in range(3)))

    def __call__(self,jets):
        if (jets.te_order+1!=self.shape[1] or jets.potential_order!=0 or
                any(np.any(getattr(jets,k)) for k in ('psi','psix','psis'))):
            raise ValueError('CACHE_JET_MODEL_MISMATCH')
        x=np.stack([getattr(jets,k) for k in FIELDS],axis=-1).ravel()/self.scale
        if not np.isfinite(x).all(): raise ValueError('CACHE_JET_NONFINITE')
        # Keep the original kernel for states outside the small-strain
        # benchmark envelope; never relax its physical validity checks.
        if np.max(np.abs(x))>50: return self.direct(jets)
        g=self.K@x
        return self.work_type(float(.5*np.dot(x,g)*self.energy_scale),
                              (g*self.energy_scale/self.scale).reshape(self.shape),
                              np.zeros((1,3)),self.gauss)


_CACHE = {}


def install_verified_linear_local(model,u0):
    # The constitutive law has no dependence on surface node count or solver.
    # Cache only the original function and all its exact physical parameters.
    from functools import partial
    original=model._local
    if not isinstance(original,partial):
        raise TypeError('CACHE_REQUIRES_ORIGINAL_LOCAL_PARTIAL')
    key=(original.func,model.theory,model.geometry_model,model.material,
         model.grid.radius_m,model.thickness_m,model.n_gauss,float(u0))
    cache=_CACHE.get(key)
    if cache is None:
        cache=VerifiedLinearCUFLocal(model,u0)
        _CACHE[key]=cache
    else:
        if cache.direct.func is not original.func or cache.direct.keywords!=original.keywords or cache.direct.args!=original.args:
            raise ValueError('CACHE_REUSE_PHYSICAL_PARAMETERS_MISMATCH')
    print('LINEAR_LOCAL_AUDIT '+json.dumps(cache.audit,sort_keys=True),flush=True)
    model._local=cache
    return cache.audit
