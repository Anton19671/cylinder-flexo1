"""Complete angular Fourier block solver for homogeneous circular modal FDM.

This is a unitary change of coordinates of the existing assembled matrix-free
K/M actions, not a replacement continuum model or a reduced-rank MVI method.
Every angular harmonic is included. All axial/component DOFs remain free.
For odd periodic grids, n>0 sectors have a real sine/cosine doublet.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh, solve, qr


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
    orth, triangular, _ = qr(traces.T, mode="full",
                             pivoting=True, check_finite=False)
    diagonal = np.abs(np.diag(triangular))
    rank = int(np.count_nonzero(
        diagonal > 1.e-11 * np.max(diagonal, initial=0.)))
    if rank != len(reference):
        return None
    return orth[:, rank:].copy()


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


def _quotient(km,mm,vector):
    kv,mv=km@vector,mm@vector
    denominator=float(np.vdot(vector,mv).real)
    numerator=float(np.vdot(vector,kv).real)
    if not np.isfinite([numerator,denominator]).all() or denominator<=0.:
        raise ValueError('MODAL_FOURIER_INVALID_RAYLEIGH_QUOTIENT')
    return numerator/denominator


def _relative_eigen_quality(kv,mv,value):
    return float(np.linalg.norm(kv-value*mv)/
                 max(np.linalg.norm(kv)+abs(value)*np.linalg.norm(mv),1.e-30))


def _refine_fourier_eigenpair(km,mm,value,vector,*,max_iterations=3):
    """Recompute actual Ritz energy, then refine the original Fourier pencil."""
    vector=np.asarray(vector,complex).copy()
    lam=_quotient(km,mm,vector)
    for iteration in range(max_iterations+1):
        quality=_relative_eigen_quality(km@vector,mm@vector,lam)
        if quality<=5.e-9: break
        if iteration==max_iterations:
            raise ValueError('MODAL_FOURIER_BLOCK_EIGENPAIR_NOT_VERIFIED')
        shift=lam-1.e-3*max(abs(lam),1.e-3)
        candidate=solve(km-shift*mm,mm@vector,assume_a='her',check_finite=False)
        energy=float(np.vdot(candidate,mm@candidate).real)
        if not np.isfinite(energy) or energy<=0.:
            raise ValueError('MODAL_FOURIER_REFINEMENT_BAD_MASS')
        vector=candidate/np.sqrt(energy)
        lam=_quotient(km,mm,vector)
    return lam,vector,quality


def _certify_full_mode(k,m,vector,value,prediction_k,prediction_m,
                       *,quality_limit=1.e-6, field_shape=None,
                       circumferential_n=None, mode_index=None, origin=None):
    """Verify physical actions and quantify numerical block-action disagreement."""
    kv,mv=k@vector,m@vector
    denominator=max(np.linalg.norm(kv)+abs(value)*np.linalg.norm(mv),1.e-30)
    physical=float(np.linalg.norm(kv-value*mv)/denominator)
    projected=float(np.linalg.norm(prediction_k-value*prediction_m)/denominator)
    mismatch=float((np.linalg.norm(kv-prediction_k)+
                    abs(value)*np.linalg.norm(mv-prediction_m))/denominator)
    mass=float(vector@mv)
    if not np.isfinite([physical,projected,mismatch,mass]).all() or mass<=0.:
        raise ValueError('MODAL_FOURIER_NONFINITE_FULL_EIGENPAIR')
    physical_quotient=float((vector@kv)/mass)
    if (projected>5.e-9 or mismatch>quality_limit or
            physical>projected+mismatch+2.e-8 or
            abs(physical_quotient-value)>5.e-8*max(1.,abs(value))):
        # The rejected pair is still a failure: report independently where
        # the physical and Fourier-predicted actions disagree, without
        # weakening the acceptance threshold or modifying K/M.
        k_delta=kv-prediction_k
        m_delta=mv-prediction_m
        k_mismatch=float(np.linalg.norm(k_delta)/denominator)
        m_mismatch=float(abs(value)*np.linalg.norm(m_delta)/denominator)
        angular_diagnostic=''
        if field_shape is not None and circumferential_n is not None:
            n=int(circumferential_n)
            ns=int(field_shape[-1])
            other=[i for i in range(ns) if i not in {n,(-n)%ns}]
            def harmonic_leakage(field):
                ft=np.fft.fft(np.asarray(field).reshape(field_shape),
                              axis=-1)
                return float(np.linalg.norm(ft[:,:,other])/
                             max(np.linalg.norm(ft),1.e-30))
            angular_diagnostic=(
                f', K_off_harmonic={harmonic_leakage(kv):.6e}'
                f', M_off_harmonic={harmonic_leakage(mv):.6e}'
                f', K_delta_off_harmonic={harmonic_leakage(k_delta):.6e}'
                f', M_delta_off_harmonic={harmonic_leakage(m_delta):.6e}')
        raise ValueError(
            'MODAL_FOURIER_EIGENPAIR_VERIFICATION_FAILED: '
            f'mode={mode_index}, n={circumferential_n}, phase={origin}, '
            f'physical={physical:.6e}, projected={projected:.6e}, '
            f'block_action_mismatch={mismatch:.6e}, '
            f'K_action_mismatch={k_mismatch:.6e}, '
            f'M_action_mismatch={m_mismatch:.6e}, '
            f'physical_rayleigh={physical_quotient:.12g}, '
            f'projected_eigenvalue={value:.12g}, '
            f'mass={mass:.6e}' + angular_diagnostic)
    return physical,projected,mismatch


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
        if not np.isfinite(values).all() or np.any(
                values < -1.e-8*max(1.,np.max(np.abs(values)))):
            raise ValueError('MODAL_FOURIER_INVALID_EIGENVALUE')
        for index in range(len(values)):
            lam,vec,quality=_refine_fourier_eigenpair(
                km,mm,values[index],vectors[:,index])
            if lam < -1.e-8*max(1.,float(np.max(np.abs(values)))):
                raise ValueError('MODAL_FOURIER_INVALID_REFINED_EIGENVALUE')
            lam=max(float(lam),0.)
            axial_shape=(axial_basis@vec).reshape(q0.shape[:2])
            shape=axial_shape[:,:,None]*phase[None,None,:]
            generation=('FOURIER_CONSTANT',) if n==0 else ('FOURIER_COS','FOURIER_SIN')
            candidates=(shape.real,) if n==0 else (
                np.sqrt(2.)*shape.real,np.sqrt(2.)*shape.imag)
            kpred_ax=(axial_basis@(km@vec)).reshape(q0.shape[:2])
            mpred_ax=(axial_basis@(mm@vec)).reshape(q0.shape[:2])
            kpred=kpred_ax[:,:,None]*phase[None,None,:]
            mpred=mpred_ax[:,:,None]*phase[None,None,:]
            pk=(kpred.real,) if n==0 else (
                np.sqrt(2.)*kpred.real,np.sqrt(2.)*kpred.imag)
            pm=(mpred.real,) if n==0 else (
                np.sqrt(2.)*mpred.real,np.sqrt(2.)*mpred.imag)
            for origin,field,pred_k,pred_m in zip(generation,candidates,pk,pm):
                entries.append((lam,n,index+1,origin,field.ravel().copy(),
                                pred_k.ravel().copy(),pred_m.ravel().copy()))
            families.append({'n':n,'axial_mode_index':index+1,
                             'axial_mode_index_verified':False,
                             'frequency_hz':float(np.sqrt(lam*conversion)/(2*np.pi)),
                             'degeneracy':len(generation)})
        if detail_callback:
            detail_callback(stage='MATRIX_FREE_MODAL_FOURIER_SECTOR_DONE',
                            circumferential_n=n,
                            first_frequency_hz=float(np.sqrt(max(_quotient(km,mm,vectors[:,0]),0.)*conversion)/(2*np.pi)))
    entries.sort(key=lambda row:(row[0],row[1],row[2],row[3]))
    chosen=entries[:requested]
    if len(chosen)<requested:
        raise ValueError('MODAL_FOURIER_NOT_ENOUGH_EIGENMODES')
    values=np.array([v[0] for v in chosen])
    vectors=np.column_stack([v[4] for v in chosen])
    for j,row in enumerate(chosen):
        _certify_full_mode(k,m,vectors[:,j],values[j],row[5],row[6],
                           field_shape=q0.shape, circumferential_n=row[1],
                           mode_index=j+1, origin=row[3])
    m_vectors=m@vectors
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
