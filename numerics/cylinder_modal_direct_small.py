"""Direct full-grid generalized modal eigenproblem on bounded small CSR grids.

Matrices are assembled from the same projected physical stiffness and
consistent mass actions as the matrix-free FDM route. No rank separation,
substitute shell theory, support penalties, or fictitious mass are introduced.
Only simple one-DOF homogeneous essential traces are supported.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def can_use_modal_direct_small(projector, *, limit=2300):
    """Constrained coordinate subspace, small enough for two dense K/M matrices."""
    if not 1 <= limit <= 2300 or projector.count > projector.size:
        return False
    c = projector.c
    if (c.getnnz(axis=1) != 1).any() or (c.getnnz(axis=0) > 1).any():
        return False
    if (np.max(abs(projector.target), initial=0.) > 1.e-12 or
            np.max(abs(projector.lift), initial=0.) > 1.e-12):
        return False
    n_free = projector.size - projector.count
    return 1 <= n_free <= limit and np.count_nonzero(c.getnnz(axis=0)==0) == n_free


def solve_modal_direct_small(prepared, density, requested, projector, *,
                             progress_callback=None, detail_callback=None):
    from cylinder_modal_matrixfree import _projected_actions
    if not can_use_modal_direct_small(projector):
        raise ValueError('MODAL_DIRECT_SMALL_REQUIRES_SIMPLE_FIXED_TRACES_AND_LIMIT')
    free = np.flatnonzero(projector.c.getnnz(axis=0)==0)
    if len(free) < requested:
        raise ValueError('MODAL_DIRECT_SMALL_INSUFFICIENT_FREE_DOFS')
    f = prepared.problem.functional
    q0 = prepared.mechanical
    k, m, scales, conversion, mass_scale = _projected_actions(
        prepared, density, projector)
    size = len(free)
    km = np.empty((size,size), dtype=float)
    mm = np.empty((size,size), dtype=float)
    trial = np.zeros(projector.size)
    if detail_callback is not None:
        detail_callback(stage='MATRIX_FREE_MODAL_DIRECT_BEGIN',
                        solver='DENSE_GENERALIZED_EIGEN', free_dofs=size,
                        memory_bytes=2*size*size*8)
    for j, index in enumerate(free):
        trial[index] = 1.
        km[:,j] = (k @ trial)[free]
        mm[:,j] = (m @ trial)[free]
        trial[index] = 0.
        if detail_callback is not None and ((j+1)%125==0 or j+1==size):
            detail_callback(stage='MATRIX_FREE_MODAL_DIRECT_COLUMN',
                            column=j+1,total_columns=size)
    for name,a in [('K',km),('M',mm)]:
        if not np.isfinite(a).all():
            raise ValueError('MODAL_DIRECT_SMALL_NONFINITE_'+name)
        asym = float(np.linalg.norm(a-a.T,ord=np.inf))/max(
            float(np.linalg.norm(a,ord=np.inf)),1.e-30)
        if asym>1.e-8:
            raise ValueError('MODAL_DIRECT_SMALL_NONSYMMETRIC_'+name)
        a[:,:] = .5*(a+a.T)
    if detail_callback is not None:
        detail_callback(stage='EIGEN_BEGIN',n_modes=requested,
                        eigen_solver='DENSE_GENERALIZED_EIGEN')
    values, vectors=eigh(km,mm,subset_by_index=(0,requested-1),
                         driver='gvx',check_finite=True)
    if not np.isfinite(values).all():
        raise ValueError('MODAL_DIRECT_SMALL_NONFINITE_EIGENVALUES')
    if np.any(values < -1.e-8*max(1.,float(np.max(np.abs(values))))):
        raise ValueError('MODAL_DIRECT_SMALL_NEGATIVE_SQUARED_FREQUENCY')
    omega = np.sqrt(np.maximum(values,0.) * conversion)
    frequencies = omega/(2*np.pi)
    fields=[]
    for j in range(requested):
        basis = np.zeros(projector.size)
        basis[free] = vectors[:,j]
        field=(scales * basis / np.sqrt(mass_scale)).reshape(q0.shape)
        fields.append(field)
        if detail_callback is not None:
            detail_callback(stage='FDM_MODE_DONE',mode=j+1,
                            n_modes=requested,frequency_hz=float(frequencies[j]),
                            omega_rad_s=float(omega[j]),
                            modal_amplitude_max=float(np.max(np.abs(field))))
        if progress_callback is not None:
            progress_callback(j+1,requested)
    return {
        'theory':f.theory,'method':'FDM','fdm_backend':'MATRIX_FREE_GMRES',
        'modal_eigensolver':'DENSE_GENERALIZED_EIGEN_SMALL_GRID',
        'density_kg_m3':float(density),
        'frequencies_hz':frequencies,
        'omega_rad_s':omega,
        'omega_squared_rad2_s2':np.maximum(values,0.)*conversion,
        'mechanical_modes':np.stack(fields),
        'electrical_modes':None,
        'n_mechanical_free_dofs':size,
        'n_electric_free_dofs':0,
    }


__all__=('can_use_modal_direct_small','solve_modal_direct_small')
