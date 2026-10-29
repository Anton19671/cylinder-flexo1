"""V-9: saved physical fields from accepted reference states, no legacy z**k."""
import json
import numpy as np
import pytest
from cylinder_reference_results import reference_catalog, reference_step_data, reference_field_2d, reference_field_3d

def sample(tmp_path, physical=True):
    nx,ns,nz=4,5,3
    z=np.array([-.1,0.,.1]);q=np.zeros((6,nx,ns));q[0]=1.;q[1]=2.;q[4]=5.;q[5]=2.
    disp=np.zeros((nz,nx,ns,3));disp[...,0]=1.;disp[...,1]=2.
    disp[...,2]=5.+(2.*z/.2)[:,None,None]*2.
    strain=np.empty((nz,nx,ns,6));stress=np.empty_like(strain)
    for i in range(6):strain[...,i]=i+3;stress[...,i]=2*i+10
    arrays=dict(mechanical=q,z_m=z,strain6=strain,stress6_pa=stress,det_F=np.ones((nz,nx,ns)))
    if physical:arrays['displacement3_m']=disp
    np.savez_compressed(tmp_path/'REFERENCE_STEP_0001.npz',**arrays)
    summary={'solver_backend':'VARIATIONAL_REFERENCE','theory':'CUF_TE1',
        'calculation_kind':'STATIC','steps':[{'index':1,'converged':True,'state_file':'REFERENCE_STEP_0001.npz'},
        {'index':2,'converged':False,'state_file':'REFERENCE_FAILED_ITERATE.npz'}]}
    (tmp_path/'FINAL_RUN_SUMMARY.json').write_text(json.dumps(summary))

def test_catalog_rejects_unaccepted(tmp_path):
    sample(tmp_path);summary,rows=reference_catalog(tmp_path)
    assert summary['theory']=='CUF_TE1' and len(rows)==1
    with pytest.raises(ValueError,match='not accepted'):reference_step_data(tmp_path,2)

def test_physical_fields_thickness_basis(tmp_path):
    sample(tmp_path);d=reference_step_data(tmp_path,1)
    assert d['w0'].shape==(5,4) and np.all(d['w0']==5.)
    z,w,_=reference_field_3d(d,'w(z)')
    assert w.shape==(3,5,4)
    np.testing.assert_array_equal(w[:,0,0],[3.,5.,7.])
    assert np.all(reference_field_2d(d,'w(z)',0.)[0]==5.)
    assert np.all(reference_field_2d(d,'S_xz',0.)[0]==18.)
    assert np.all(reference_field_2d(d,'E_xs',0.)[0]==6.)

def test_missing_fields_fail_closed(tmp_path):
    sample(tmp_path,physical=False)
    with pytest.raises(ValueError,match='missing displacement3_m'):reference_step_data(tmp_path,1)

def test_wrong_backend_fails(tmp_path):
    (tmp_path/'FINAL_RUN_SUMMARY.json').write_text(json.dumps({'solver_backend':'LEGACY','steps':[]}))
    with pytest.raises(ValueError,match='Not a VARIATIONAL_REFERENCE'):reference_catalog(tmp_path)

def test_missing_accepted_state_not_counted(tmp_path):
    sample(tmp_path);(tmp_path/'REFERENCE_STEP_0001.npz').unlink()
    assert reference_catalog(tmp_path)[1]==[]
