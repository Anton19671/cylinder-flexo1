"""Frozen-coefficient coupled shell symbol from the Krysko--Zhong enthalpy.

For local surface weak jets A_(field,alpha) = dL/dq_(field,alpha),
H_(field,alpha;field,beta) = dA_(field,alpha)/dq_(field,beta).
The full (not just top-degree) Fourier symbol of the frozen linearization is
    P_ab(k) = sum_{alpha,beta} (-i*k)^alpha H_(a,alpha;b,beta) (i*k)^beta.
Here k=(kx,ks), s=R*theta.  Mixed jets q_xs occur ONCE.  Electric
potential modes use independent 0/x/s jets and have no electric inertia.

A coupled mixed-order principal symbol cannot be classified by taking the
maximum stored jet order (KL/FSDT/Reddy 3; CUF 2).  The homogeneous degree
matrices and pairwise degree support provided here are DIAGNOSTICS.  They do
not certify Douglis--Nirenberg orders, ellipticity, strong boundary closures,
or an independent strong FDM solver.  No matrix inversion, SVD or solver run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import numpy as np

MECHANICAL_ORDERS = (
    (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),
    (3,0),(2,1),(1,2),(0,3),
)
ELECTRICAL_ORDERS = ((0,0),(1,0),(0,1))
THEORIES = ("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4")


def _validate_orders(orders, name):
    result=tuple(orders)
    if not result or any(not isinstance(x,(tuple,list)) or len(x)!=2 or
                         any(type(v)!=int or v<0 for v in x) or
                         sum(x)>3 for x in result) or len(set(map(tuple,result)))!=len(result):
        raise ValueError(name+": unique nonnegative surface orders through three required")
    if tuple(result[0])!=(0,0):
        raise ValueError(name+": base 0,0 jet is required")
    return tuple(map(tuple,result))


@dataclass(frozen=True)
class CoupledJetLayout:
    """Independent DOFs grouped field-major, then derivative-jet-major."""
    n_mechanical: int
    n_electric: int
    mechanical_orders: tuple
    electric_orders: tuple=ELECTRICAL_ORDERS

    def __post_init__(self):
        for label,n in (("mechanical",self.n_mechanical),("electric",self.n_electric)):
            if type(n)!=int or n<(1 if label=="mechanical" else 0):
                raise ValueError(label+": invalid number of generalized fields")
        object.__setattr__(self,"mechanical_orders",_validate_orders(self.mechanical_orders,"mechanical"))
        object.__setattr__(self,"electric_orders",_validate_orders(self.electric_orders,"electric"))

    @property
    def n_fields(self):
        return self.n_mechanical+self.n_electric

    @property
    def n_jets(self):
        return self.n_mechanical*len(self.mechanical_orders)+self.n_electric*len(self.electric_orders)

    @property
    def axes(self):
        return ([(field,ab) for field in range(self.n_mechanical)
                 for ab in self.mechanical_orders] +
                [(self.n_mechanical+field,ab) for field in range(self.n_electric)
                 for ab in self.electric_orders])

    @classmethod
    def for_theory(cls, theory, n_electric=0):
        if theory not in THEORIES:
            raise ValueError("theory must be KL/FSDT/REDDY/CUF_TE1..TE4")
        n=3 if theory=="KL" else 5 if theory in ("FSDT","REDDY") else 3*(int(theory[-1])+1)
        return cls(n,n_electric,MECHANICAL_ORDERS[:10 if theory in ("KL","FSDT","REDDY") else 6])


@dataclass(frozen=True)
class CoupledFrozenSymbol:
    layout: CoupledJetLayout
    jet_tangent: np.ndarray
    reciprocity_rel_error: float
    source: str="KRYŚKO_ZHONG_FROZEN_LOCAL_ENTHALPY"
    status: str="COUPLED_SYMBOL_DIAGNOSTIC_NOT_STRONG_ORDER_CERTIFICATE"

    def homogeneous_degrees(self, kx, ks):
        """P(t*kx,t*ks)=sum_d t**d P_d(kx,ks), including all cross blocks.

        Multipliers use the exact continuum Euler adjoint (-i*k)^alpha;
        this is NOT the symbol of a selected finite-difference stencil.
        """
        kx,ks=float(kx),float(ks)
        if not math.isfinite(kx) or not math.isfinite(ks) or (kx==0 and ks==0):
            raise ValueError("finite nonzero physical wave vector required")
        axes=self.layout.axes
        result={d:np.zeros((self.layout.n_fields,self.layout.n_fields),complex)
                for d in range(7)}
        phases=np.array([(1j*kx)**ax*(1j*ks)**bs for _,(ax,bs) in axes])
        degrees=np.array([sum(ab) for _,ab in axes],dtype=int)
        for j,(field_j,_) in enumerate(axes):
            if phases[j]==0:continue
            for i,(field_i,_) in enumerate(axes):
                if phases[i]==0:continue
                result[int(degrees[i]+degrees[j])][field_i,field_j] += (
                    np.conjugate(phases[i])*self.jet_tangent[i,j]*phases[j])
        return {k:v for k,v in result.items() if np.any(v!=0)}

    def evaluate(self,kx,ks):
        """Full frozen-coefficient Fourier symbol, mechanical/electric saddle."""
        values=self.homogeneous_degrees(kx,ks)
        return sum(values.values(),np.zeros((self.layout.n_fields,self.layout.n_fields),complex))

    def directional_coupling_audit(self, directions=((1.,0.),(0.,1.),(1.,1.)),
                                   *,relative_cutoff=1.e-8):
        """Inspect every mechanical/electric block on several k directions.

        Pairwise degrees are not Douglis--Nirenberg weights and may change
        with direction. Returns upper bounds on observed polynomial powers,
        plus whether high-order mechanical/electrical cross terms are present.
        No claim that a boundary-value problem is well posed follows.
        """
        if not directions:
            raise ValueError("at least one physical wave direction required")
        tables=[self.pairwise_highest_degrees(x,s,relative_cutoff=relative_cutoff)
                for x,s in directions]
        maximum=np.maximum.reduce(tables)
        m=self.layout.n_mechanical
        return {
            "directions":tuple(tuple(map(float,item)) for item in directions),
            "degrees_by_direction":tuple(tables),
            "maximum_observed_degrees":maximum,
            "mechanical_electric_coupling":bool(m<self.layout.n_fields and
                (np.any(maximum[:m,m:]>=0) or np.any(maximum[m:,:m]>=0))),
            "status":"MIXED_COUPLED_SYMBOL_ORDER_NOT_CERTIFIED",
        }

    def pairwise_highest_degrees(self,kx,ks,*,relative_cutoff=1.e-8):
        """Highest observed degree for each field pair, NOT a BC order.

        Directional cancellation is possible: inspect multiple k directions.
        Physical units can differ between blocks; each matrix entry uses its
        own polynomial-degree maximum, not a single combined field scale.
        """
        if isinstance(relative_cutoff,(bool,np.bool_)) or not (0<relative_cutoff<1):
            raise ValueError("relative_cutoff must be in (0,1)")
        hom=self.homogeneous_degrees(kx,ks)
        table=np.full((self.layout.n_fields,self.layout.n_fields),-1,int)
        if not hom:
            return table
        scales=np.maximum.reduce([np.abs(v) for v in hom.values()])
        for degree,mat in hom.items():
            supported=(np.abs(mat)>relative_cutoff*scales)&(scales>0)
            table[supported]=np.maximum(table[supported],degree)
        return table


def frozen_symbol_from_tangent(layout, jet_tangent, *, reciprocity_tolerance=1e-5):
    """Build entire mechanical/electric polynomial without dropping mixed jets."""
    if not isinstance(layout,CoupledJetLayout):
        raise TypeError("CoupledJetLayout required")
    h=np.asarray(jet_tangent,float)
    n=layout.n_jets
    if h.shape!=(n,n) or not np.isfinite(h).all():
        raise ValueError("finite full local jet tangent must be square")
    if isinstance(reciprocity_tolerance,(bool,np.bool_)) or not 0<reciprocity_tolerance<1:
        raise ValueError("reciprocity tolerance must be in (0,1)")
    # Check blockwise scaled symmetry; never overwrite an asymmetric tangent
    # with (H+H.T)/2 (that would conceal an incorrect G3D derivative).
    def metric(a,b):
        if a.size==0:return 0.
        scale=max(1.e-30,float(np.max(np.abs(a))),float(np.max(np.abs(b))))
        return float(np.max(np.abs(a-b)))/scale
    im=layout.n_mechanical*len(layout.mechanical_orders)
    dev=max(metric(h[:im,:im],h[:im,:im].T),
            metric(h[im:,im:],h[im:,im:].T),
            metric(h[:im,im:],h[im:,:im].T))
    if dev>reciprocity_tolerance:
        raise ValueError("COUPLED_LOCAL_JET_RECIPROCITY_MISMATCH")
    return CoupledFrozenSymbol(layout,h.copy(),dev)


def finite_difference_local_jet_tangent(layout, weak_gradient: Callable,
                                        *, state=None, jet_scales=None,
                                        relative_step=1e-5,
                                        reciprocity_tolerance=2e-4):
    """Differentiate existing local enthalpy *first variations* only.

    weak_gradient(flat_jets) is the unweighted local Gateaux gradient in
    the same (field-major,jet-major) ordering. Never finite-difference
    the assembled reference FDM or introduce a second constitutive model.
    The scales are physically dimensioned per jet and MUST be supplied for
    nonzero states where field/derivative units are disparate.
    """
    if not isinstance(layout,CoupledJetLayout) or not callable(weak_gradient):
        raise TypeError("layout and local weak-gradient callable required")
    n=layout.n_jets
    x=np.zeros(n,float) if state is None else np.asarray(state,float)
    scale=np.ones(n,float) if jet_scales is None else np.asarray(jet_scales,float)
    if x.shape!=(n,) or scale.shape!=(n,) or not np.isfinite(x).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError("state and positive physical jet scales must have n_jets entries")
    if jet_scales is None and np.any(x!=0):
        raise ValueError("nonzero physical jets require explicit dimensioned jet_scales")
    if isinstance(relative_step,(bool,np.bool_)) or not 0<relative_step<.01:
        raise ValueError("relative_step must be in (0,0.01)")
    def evaluate(z):
        value=np.asarray(weak_gradient(z),float)
        if value.shape!=(n,) or not np.isfinite(value).all():
            raise ValueError("local enthalpy variation has invalid size or values")
        return value
    evaluate(x)
    h=np.empty((n,n),float)
    for j in range(n):
        increment=relative_step*scale[j]
        delta=np.zeros(n);delta[j]=increment
        h[:,j]=(evaluate(x+delta)-evaluate(x-delta))/(2*increment)
    return frozen_symbol_from_tangent(layout,h,reciprocity_tolerance=reciprocity_tolerance)


def local_shell_symbol(theory,material,*,radius_m,thickness_m,
                       potential_order=0,n_gauss=4,mechanical_jets=None,
                       electric_jets=None,jet_scales=None,relative_step=1e-5,
                       reciprocity_tolerance=2e-4):
    """Reference seven-theory G3D/Gz symbol, using ORIGINAL thickness kernels.

    Does not create a grid, solve equilibrium, time-step, or choose BC orders.
    The caller must choose a physically admissible local state and scales.
    An undeformed zero state is used only when no jets are supplied.
    """
    from flexo_gradient_g3d import G3DMaterial
    if not isinstance(material,G3DMaterial):
        raise TypeError("actual G3DMaterial (G3D or GZ) required")
    if theory not in THEORIES:
        raise ValueError("unknown cylinder theory")
    if type(potential_order)!=int or potential_order<0 or potential_order>4:
        raise ValueError("potential_order must be integer 0..4")
    if type(n_gauss)!=int or n_gauss<2:
        raise ValueError("n_gauss must be integer >=2")
    R,h=float(radius_m),float(thickness_m)
    if not (math.isfinite(R) and math.isfinite(h) and R>h/2>0):
        raise ValueError("require R>h/2>0")
    active=(material.kappa_f_per_m>0)
    layout=CoupledJetLayout.for_theory(theory,potential_order+1 if active else 0)
    mech=np.zeros((layout.n_mechanical,len(layout.mechanical_orders))) if mechanical_jets is None else np.asarray(mechanical_jets,float)
    electric_shape=(layout.n_electric,len(layout.electric_orders))
    elec=np.zeros(electric_shape) if electric_jets is None else np.asarray(electric_jets,float)
    if mech.shape!=(layout.n_mechanical,len(layout.mechanical_orders)) or not np.isfinite(mech).all():
        raise ValueError("mechanical_jets must be finite component-major local jets")
    if elec.shape!=electric_shape or not np.isfinite(elec).all():
        raise ValueError("electric_jets must contain exactly the enabled potential modes")
    if theory=="KL":
        from cylinder_kl_g3d_weak import KLSurfaceJets as Jet, thickness_weak_coefficients as kernel
        def evaluate(m,e):
            jets=Jet(*[m[:,j] for j in range(m.shape[1])],
                     *[e[:,j] for j in range(e.shape[1])])
            return kernel(jets,material,radius_m=R,thickness_m=h,n_gauss=n_gauss)
    elif theory in ("FSDT","REDDY"):
        from cylinder_director_g3d import DirectorSurfaceJets as Jet, thickness_weak_coefficients as kernel
        def evaluate(m,e):
            jets=Jet(*[m[:,j] for j in range(m.shape[1])],
                     *[e[:,j] for j in range(e.shape[1])])
            return kernel(jets,material,theory=theory,radius_m=R,thickness_m=h,n_gauss=n_gauss)
    else:
        from cuf_g3d_weak import CUFSurfaceJets as Jet, thickness_weak_coefficients as kernel
        n=int(theory[-1])+1
        def evaluate(m,e):
            jets=Jet(*[m[:,j].reshape((3,n)) for j in range(m.shape[1])],
                     *[e[:,j] for j in range(e.shape[1])])
            return kernel(jets,material,radius_m=R,thickness_m=h,n_gauss=n_gauss)
    nm=mech.size
    state=np.concatenate((mech.ravel(),elec.ravel()))
    def weak_gradient(z):
        m=z[:nm].reshape(mech.shape)
        e=z[nm:].reshape(elec.shape) if active else np.zeros((1,3))
        result=evaluate(m,e)
        mout=np.asarray(result.mechanical,float).reshape(mech.shape)
        if active:
            pout=np.asarray(result.electrical,float).reshape(elec.shape)
            return np.concatenate((mout.ravel(),pout.ravel()))
        return mout.ravel()
    return finite_difference_local_jet_tangent(
        layout,weak_gradient,state=state,jet_scales=jet_scales,
        relative_step=relative_step,reciprocity_tolerance=reciprocity_tolerance)


__all__=("MECHANICAL_ORDERS","ELECTRICAL_ORDERS","THEORIES",
         "CoupledJetLayout","CoupledFrozenSymbol","frozen_symbol_from_tangent",
         "finite_difference_local_jet_tangent","local_shell_symbol")
