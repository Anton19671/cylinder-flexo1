"""Complete angular Fourier block solver for homogeneous circular modal FDM.

This is a unitary change of coordinates of the existing assembled matrix-free
K/M actions, not a replacement continuum model or a reduced-rank MVI method.
Every angular harmonic is included. All axial/component DOFs remain free.
For odd periodic grids, n>0 sectors have a real sine/cosine doublet.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh


def _axial_constraint_basis(prepared, projector):
    """Obtain the exact axial/component null space for periodic-uniform traces.

    Each physical boundary row must act within one circumferential node,
    and every node must carry the same axial constraints. KL clamped-end
    slopes use multi-node stencils, so merely deleting their touched DOFs
    would change the boundary condition. The same null space is reused
    in every Fourier sector instead.
    """
    q = prepared.mechanical
    ncomp, nx, ns = q.shape
    n_axial = ncomp * nx
    c = projector.c.tocsr()
    if c.shape != (projector.count, q.size):
        return None
    rows = [[] for _ in range(ns)]
    for row in range(c.shape[0]):
        start, stop = c.indptr[row:row + 2]
        indices, values = c.indices[start:stop], c.data[start:stop]
        if not len(indices) or np.any(indices % ns != indices[0] % ns):
            return None
        axial = np.zeros(n_axial)
        axial[indices // ns] = values
        first = np.flatnonzero(np.abs(axial) > 1.e-14)
        if not len(first):
            return None
        axial /= np.linalg.norm(axial)
        if axial[first[0]] < 0:
            axial = -axial
        rows[int(indices[0] % ns)].append(axial)
    def canonical(entries):
        return sorted(entries, key=lambda item: tuple(np.round(item, 12)))
    reference = canonical(rows[0])
    for other in rows[1:]:
        candidate = canonical(other)
        if (len(candidate) != len(reference) or
                any(not np.allclose(a, b, rtol=1.e-10, atol=1.e-12)
                    for a, b in zip(reference, candidate))):
            return None
    if not reference:
        return np.eye(n_axial)
    traces = np.stack(reference)
    touches = np.count_nonzero(np.abs(traces) > 1.e-14, axis=1)
    if np.all(touches == 1):
        fixed = np.flatnonzero(np.any(np.abs(traces) > 1.e-14, axis=0))
        if len(fixed) != len(reference):
            return None
        return np.eye(n_axial)[:, np.setdiff1d(np.arange(n_axial), fixed)]
    _, singular, vt = np.linalg.svd(traces, full_matrices=True)
    rank = int(np.count_nonzero(singular > 1.e-11 * singular[0]))
    if rank != len(reference):
        return None
    return vt[rank:].T.copy()


def can_use_fourier_modal_fdm(prepared, projector):
    """Allow only demonstrably Fourier-invariant, simple homogeneous traces."""
    f=prepared.problem.functional
    q=prepared.mechanical
    if (prepared.method!='FDM' or prepared.fdm_backend!='MATRIX_FREE_GMRES'
            or prepared.electrical is not None or f.geometry_model!='LINEAR'
            or f.material.ell_m or q.ndim!=3):
        return False
    nx,ns=q.shape[1:]
    if nx<9 or ns<9 or ns%2!=1:
        return False
    if (np.max(np.abs(projector.target),initial=0.)>=1e-12
            or np.max(np.abs(projector.lift),initial=0.)>=1e-12):
        return False
    axial_basis = _axial_constraint_basis(prepared, projector)
    if axial_basis is None or not 3 <= axial_basis.shape[1] <= 900:
        return False
    ops=f.ops
    if not np.allclose(ops.ws,ops.ws[0],rtol=1e-12,atol=1e-14):
        return False
    if not np.isclose(ops.wx[0]/ops.wx[4],17./48.,rtol=1e-11,atol=1e-14):
        return False
    theta=2*np.pi*np.arange(ns)/ns
    for n in (1,(ns-1)//2):
        phase=np.exp(1j*n*theta)/np.sqrt(ns)
        true=1j*n/f.R*phase
        derivative=ops.Ds@phase
        if np.linalg.norm(derivative-true)>1e-9*max(np.linalg.norm(true),1e-30):
            return False
    return True


def solve_modal_fourier_fdm(prepared,density,requested,projector,
                            k,m,scales,conversion,mass_scale,*,
                            progress_callback=None,detail_callback=None):
    """Find the lowest global discrete eigenpairs over ALL angular sectors."""
    if not can_use_fourier_modal_fdm(prepared,projector):
        raise ValueError('MODAL_FOURIER_REQUIRES_UNIFORM_CIRCULAR_SBP42_CSR')
    q0=prepared.mechanical
    f=prepared.problem.functional
    nx,ns=q0.shape[1:]
    axial_basis = _axial_constraint_basis(prepared, projector)
    if axial_basis is None:
        raise ValueError('MODAL_FOURIER_AXIAL_TRACE_SPACE_MISMATCH')
    count = axial_basis.shape[1]
    theta=2.*np.pi*np.arange(ns)/ns
    entries=[]
    families=[]
    for n in range((ns-1)//2+1):
        if detail_callback:
            detail_callback(stage='MATRIX_FREE_MODAL_FOURIER_SECTOR_BEGIN',
                            circumferential_n=n,
                            total_sectors=(ns+1)//2,sector_dofs=count)
        phase=np.exp(1j*n*theta)/np.sqrt(ns)
        conjugate=np.conjugate(phase)
        factor=2. if n else 1.
        km=np.empty((count,count),dtype=np.complex128)
        mm=np.empty((count,count),dtype=np.complex128)
        trial=np.zeros_like(q0)
        for j in range(count):
            trial[:, :, :] = (axial_basis[:, j].reshape(q0.shape[:2])[:, :, None]
                              * phase.real[None, None, :])
            ka=(k@trial.ravel()).reshape(q0.shape)
            ma=(m@trial.ravel()).reshape(q0.shape)
            for label,action in (('K',ka),('M',ma)):
                spectrum=np.fft.fft(action,axis=-1)/np.sqrt(ns)
                other=np.array([idx for idx in range(ns)
                                if idx not in {n,(-n)%ns}],dtype=int)
                leakage=(np.linalg.norm(spectrum[:,:,other])/
                         max(np.linalg.norm(spectrum),1e-30))
                if leakage>1e-8:
                    raise ValueError('MODAL_FOURIER_'+label+'_BREAKS_ANGULAR_SYMMETRY')
            k_axial=np.einsum('cxs,s->cx',ka,conjugate).reshape(-1)
            m_axial=np.einsum('cxs,s->cx',ma,conjugate).reshape(-1)
            km[:,j]=factor*(axial_basis.T@k_axial)
            mm[:,j]=factor*(axial_basis.T@m_axial)
            if detail_callback and ((j+1)%100==0 or j+1==count):
                detail_callback(stage='MATRIX_FREE_MODAL_FOURIER_COLUMN',
                                circumferential_n=n,column=j+1,total_columns=count)
        for label,mat in (('K',km),('M',mm)):
            if not np.isfinite(mat).all():
                raise ValueError('MODAL_FOURIER_NONFINITE_'+label)
            nonhermitian=(np.linalg.norm(mat-mat.conj().T)/
                          max(np.linalg.norm(mat),1e-30))
            if nonhermitian>1e-8:
                raise ValueError('MODAL_FOURIER_NONHERMITIAN_'+label)
            mat[:,:]=.5*(mat+mat.conj().T)
        upto=min(requested,count)-1
        values,vectors=eigh(km,mm,subset_by_index=(0,upto),driver='gvx',
                            check_finite=True)
        if not np.isfinite(values).all() or np.any(values < -1e-8*max(1.,np.max(np.abs(values)))):
            raise ValueError('MODAL_FOURIER_INVALID_EIGENVALUE')
        small=1.e-12*max(float(np.max(np.abs(values),initial=0.)),1.e-30)
        values[np.abs(values)<=small]=0.
        for index, lam in enumerate(values):
            lam=max(float(lam),0.)
            axial_shape=(axial_basis@vectors[:,index]).reshape(q0.shape[:2])
            shape=axial_shape[:,:,None]*phase[None,None,:]
            generation=('FOURIER_CONSTANT',) if n==0 else ('FOURIER_COS','FOURIER_SIN')
            candidates=(shape.real,) if n==0 else (
                np.sqrt(2.)*shape.real,np.sqrt(2.)*shape.imag)
            for origin,scaled_field in zip(generation,candidates):
                entries.append((lam,n,index+1,origin,scaled_field.ravel().copy()))
            families.append({'n':n,'axial_mode_index':index+1,
                             'axial_mode_index_verified':False,
                             'frequency_hz':float(np.sqrt(lam*conversion)/(2*np.pi)),
                             'degeneracy':len(generation)})
        if detail_callback:
            detail_callback(stage='MATRIX_FREE_MODAL_FOURIER_SECTOR_DONE',
                            circumferential_n=n,
                            first_frequency_hz=float(np.sqrt(max(values[0],0.)*conversion)/(2*np.pi)))
    entries.sort(key=lambda row:(row[0],row[1],row[2],row[3]))
    chosen=entries[:requested]
    if len(chosen)<requested:
        raise ValueError('MODAL_FOURIER_NOT_ENOUGH_EIGENMODES')
    values=np.array([v[0] for v in chosen])
    vectors=np.column_stack([v[4] for v in chosen])
    k_vectors=k@vectors
    m_vectors=m@vectors
    error=k_vectors-m_vectors*values[None,:]
    relative=np.linalg.norm(error,axis=0)/np.maximum(
        np.linalg.norm(k_vectors,axis=0)+np.abs(values)*np.linalg.norm(m_vectors,axis=0),1e-14)
    # Rigid-body modes can have eigenvalues at roundoff while the
    # nonzero spectrum is well separated. Verify such zero modes using
    # the overall spectral scale; the ordinary relative check becomes
    # meaningless when both stiffness action and eigenvalue vanish.
    largest = max(float(np.max(np.abs(values),initial=0.)),1.e-30)
    zero_mask = np.abs(values) <= 1.e-12 * largest
    near_zero_ok = (np.linalg.norm(error,axis=0) <=
                    2.e-10 * largest * np.linalg.norm(m_vectors,axis=0))
    if np.any((relative > 2.e-7) & ~(zero_mask & near_zero_ok)):
        raise ValueError('MODAL_FOURIER_EIGENPAIR_VERIFICATION_FAILED')
    values[zero_mask & near_zero_ok] = 0.
    gram=vectors.T@m_vectors
    if np.max(np.abs(gram-np.eye(requested)))>1e-6:
        raise ValueError('MODAL_FOURIER_MODES_NOT_MASS_ORTHOGONAL')
    omega=np.sqrt(values*conversion)
    freq=omega/(2*np.pi)
    modes=np.array([((scales*vectors[:,i])/np.sqrt(mass_scale)).reshape(q0.shape)
                    for i in range(requested)])
    for i in range(requested):
        if detail_callback:
            detail_callback(stage='FDM_MODE_DONE',mode=i+1,n_modes=requested,
                            circumferential_n=int(chosen[i][1]),
                            frequency_hz=float(freq[i]),
                            omega_rad_s=float(omega[i]),
                            modal_amplitude_max=float(np.max(np.abs(modes[i]))))
        if progress_callback:
            progress_callback(i+1,requested)
    next_mode=entries[requested] if len(entries)>requested else None
    cutoff_pair_incomplete=bool(next_mode is not None and
        chosen[-1][0]==next_mode[0] and
        chosen[-1][1:3]==next_mode[1:3])
    return {'theory':f.theory,'method':'FDM',
            'fdm_backend':'MATRIX_FREE_GMRES',
            'modal_eigensolver':'FULL_FOURIER_BLOCK_GENERALIZED_EIGEN',
            'density_kg_m3':float(density),
            'frequencies_hz':freq,
            'omega_rad_s':omega,
            'omega_squared_rad2_s2':values*conversion,
            'mechanical_modes':modes,'electrical_modes':None,
            'n_mechanical_free_dofs':q0.size-projector.count,
            'n_electric_free_dofs':0,
            'circumferential_wavenumbers':[int(row[1]) for row in chosen],
            'axial_mode_indices':[int(row[2]) for row in chosen],
            'harmonic_family_scan':families,
            'mode_generation':[str(row[3]) for row in chosen],
            'modal_spectrum_selection':'all_fourier_sectors_full_axial_component_space',
            'global_spectrum_completeness_verified':True,
            'cutoff_rotational_pair_incomplete':cutoff_pair_incomplete}


__all__=('can_use_fourier_modal_fdm','solve_modal_fourier_fdm')
