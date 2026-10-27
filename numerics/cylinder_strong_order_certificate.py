"""Principal-order gate for independent strong FDM, Krysko-Zhong.

Maximum surface jet degree is NOT the differential order of each field.
Classification of a coupled system requires examining its principal symbol.
This module supplies a rigorous structural preflight for independent
principal symbol matrices at each homogeneous surface degree. It must never
guess a component order from KL/FSDT/Reddy/CUF names or G3D/Gz labels.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PrincipalOrderCertificate:
    orders: tuple[int, ...]
    rank_by_degree: dict
    admissible: bool
    reason: str

def certify_decoupled_axial_orders(principal_matrices, *, rtol=1e-10):
    """Certify uncoupled pure axial principal orders 2,4,6 from symbols.

    principal_matrices maps even PDE orders 2/4/6 to square matrices
    of cross-field coefficients. Requires a permutation-free diagonal
    principal block; mixed-field orders require independent pencil
    and Douglis-Nirenberg analysis and are explicitly refused.
    """
    if not isinstance(principal_matrices,dict) or not principal_matrices:
        raise ValueError("explicit nonempty principal symbol matrices required")
    if set(principal_matrices)-{2,4,6}:
        raise ValueError("only physical axial orders 2,4,6 supported")
    mats={k:np.asarray(v,float) for k,v in principal_matrices.items()}
    n=mats[next(iter(mats))].shape[0]
    if n<1 or any(m.shape!=(n,n) or not np.isfinite(m).all() for m in mats.values()):
        raise ValueError("finite square principal matrices of equal size required")
    if not np.isfinite(rtol) or rtol<=0 or rtol>=1:
        raise ValueError("rtol must be in (0,1)")
    ranks={}
    for degree,mat in mats.items():
        # QR is sufficient for a structural matrix diagnostic; no SVD.
        # Do not apply this to certify arbitrary coupled principal pencils.
        diagonal=np.diag(mat)
        scale=max(1.,float(np.max(abs(mat))))
        off=mat-np.diag(diagonal)
        if np.max(abs(off))>rtol*scale:
            return PrincipalOrderCertificate((),{},False,
                "MIXED_PRINCIPAL_SYMBOL_REQUIRES_COUPLED_ORDER_ANALYSIS")
        ranks[degree]=int(np.count_nonzero(abs(diagonal)>rtol*scale))
    orders=[]
    for i in range(n):
        active=[degree for degree,mat in mats.items() if abs(mat[i,i])>rtol*max(1.,float(np.max(abs(mat))))]
        if not active:
            return PrincipalOrderCertificate((),ranks,False,
                "UNCLASSIFIED_COMPONENT_PRINCIPAL_ORDER")
        orders.append(max(active)//2)
    return PrincipalOrderCertificate(tuple(orders),ranks,True,
        "UNCOUPLED_AXIAL_PRINCIPAL_ORDERS_ONLY")

__all__=("PrincipalOrderCertificate","certify_decoupled_axial_orders")
