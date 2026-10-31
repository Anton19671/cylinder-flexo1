"""Complete angular Fourier block solver for homogeneous circular modal FDM.

This is a unitary change of coordinates of the existing assembled matrix-free
K/M actions, not a replacement continuum model or a reduced-rank MVI method.
Every angular harmonic is included. All axial/component DOFs remain free.
For odd periodic grids, n>0 sectors have a real sine/cosine doublet.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh


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
    c=projector.c
    if not (projector.count and (c.getnnz(axis=1)==1).all()
            and (c.getnnz(axis=0)<=1).all()
            and np.max(np.abs(projector.target),initial=0.)<1e-12
            and np.max(np.abs(projector.lift),initial=0.)<1e-12):
        return False
    free=(c.getnnz(axis=0)==0).reshape(q.shape)
    if not np.all(free==free[:,:,:1]):
        return False
    n_axial=int(np.count_nonzero(free[:,:,0]))
    if n_axial<3 or n_axial>900:
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
    mask=(projector.c.getnnz(axis=0)==0).reshape(q0.shape)
    component_axial=np.argwhere(mask[:,:,0])
    count=len(component_axial)
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
        for j,(component,axial) in enumerate(component_axial):
            trial[component,axial,:]=phase.real
            ka=(k@trial.ravel()).reshape(q0.shape)
            ma=(m@trial.ravel()).reshape(q0.shape)
            trial[component,axial,:]=0.
            if j in (0,count-1):
                for label,action in (('K',ka),('M',ma)):
                    spectrum=np.fft.fft(action,axis=-1)/np.sqrt(ns)
                    other=np.array([idx for idx in range(ns)
                                    if idx not in {n,(-n)%ns}],dtype=int)
                    leakage=(np.linalg.norm(spectrum[:,:,other])/
                             max(np.linalg.norm(spectrum),1e-30))
                    if leakage>1e-8:
                        raise ValueError('MODAL_FOURIER_'+label+'_BREAKS_ANGULAR_SYMMETRY')
            km[:,j]=factor*np.einsum('ij,j->i',ka[mask].reshape(count,ns),conjugate)
            mm[:,j]=factor*np.einsum('ij,j->i',ma[mask].reshape(count,ns),conjugate)
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
        for index, lam in enumerate(values):
            lam=max(float(lam),0.)
            shape=np.zeros(q0.shape,dtype=np.complex128)
            shape[component_axial[:,0],component_axial[:,1],:]=(
                vectors[:,index,None]*phase[None,:])
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
    if np.any(relative>2e-7):
        raise ValueError('MODAL_FOURIER_EIGENPAIR_VERIFICATION_FAILED')
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
