"""No-solve contracts for the 252-case native gradient-flexoelectric campaign."""
import hashlib
import json

import numpy as np
import pytest

from ci import native_gradient_fdm_mvi_campaign as campaign


def test_complete_matrix_is_unique_and_has_252_cases():
    cases=list(campaign.all_cases())
    assert len(cases)==7*2*2*9==252
    assert len({campaign.case_id(c) for c in cases})==252
    assert sum(campaign.included("pilot",c) for c in cases)==16


@pytest.mark.parametrize("theory",campaign.THEORIES)
@pytest.mark.parametrize("gradient",campaign.GRADIENTS)
@pytest.mark.parametrize("kind",campaign.KINDS)
def test_one_continuum_passport_for_nine_numerical_methods(theory,gradient,kind):
    physics,grid,force,passport,digest=campaign.physical_data(theory,gradient,kind)
    assert physics["material"]["ell_m"]>0
    assert physics["material"]["mu1_c_per_m"]!=0
    assert physics["material"]["mu2_c_per_m"]!=0
    assert physics["material"]["kappa_f_per_m"]>0
    assert force.shape==(campaign.nmechanical(theory),*campaign.GRID)
    if theory=="KL":
        np.testing.assert_array_equal(force,0.)
        pressure=np.asarray(passport["kl_radial_face_traction"]["traction_pa"])
        assert not np.allclose(pressure[:,0],pressure[:,2])
    else:
        assert not np.allclose(force[:,:,0],force[:,:,2])
    assert len(digest)==64
    for method,rank in (("FDM",None),)+tuple(("MVI",r) for r in range(1,9)):
        assert campaign.physical_data(theory,gradient,kind)[-1]==digest
        assert campaign.case_id((theory,gradient,kind,method,rank))


@pytest.mark.parametrize("theory",campaign.THEORIES)
@pytest.mark.parametrize("gradient",campaign.GRADIENTS)
@pytest.mark.parametrize("kind",campaign.KINDS)
def test_fdm_and_all_mvi_ranks_have_distinct_inputs_but_identical_physics(
    tmp_path,theory,gradient,kind):
    items=[(theory,gradient,kind,"FDM",None)]
    items += [(theory,gradient,kind,"MVI",r) for r in range(1,9)]
    hashes=set()
    for item in items:
        folder=tmp_path/campaign.case_id(item)
        path,forcing,passport_hash=campaign.write_input(folder,item)
        case=json.loads(path.read_text())
        hashes.add(passport_hash)
        assert case["physics"]["geometry_model"]=="GREEN_LAGRANGE"
        assert case["grid"]["nx"]==campaign.GRID[0]
        assert case["grid"]["ns"]==campaign.GRID[1]
        assert case["solver"]["kind"]==kind
        if theory=="KL":
            assert case["kl_radial_face_traction"]["face"]=="OUTER"
        if item[3]=="MVI":
            assert case["rank"]==item[4]
            with np.load(folder/"initial.npz",allow_pickle=False) as z:
                assert z["y_profiles"].shape[1]==item[4]
                assert z["x_profiles"].shape[1]==item[4]
            with np.load(folder/"loads.npz",allow_pickle=False) as z:
                np.testing.assert_array_equal(z["mechanical_load"],forcing)
        else:
            assert "rank" not in case
            expected={
                "KL":[2,2,3],"FSDT":[2]*5,"REDDY":[2,2,3,2,2],
            }.get(theory,[2]*campaign.nmechanical(theory))
            if gradient=="Gz":
                expected=[m-1 for m in expected]
            assert case["boundary"]["axial_orders"]==expected
            with np.load(folder/"loads.npz",allow_pickle=False) as z:
                np.testing.assert_array_equal(z["mechanical_load"],forcing)
    assert len(hashes)==1


@pytest.mark.parametrize("rank",range(1,9))
def test_rank_keeps_every_fourier_partner_not_replaced(rank):
    partner=campaign.fourier_seed(rank,campaign.GRID[1])
    assert partner.shape==(rank,campaign.GRID[1])
    assert np.linalg.matrix_rank(partner,tol=1.e-12)==rank



def test_comparison_separates_full_and_projected_and_rejects_stale_solver_source(tmp_path):
    """No solve: synthetic archived states test only report classification."""
    root=tmp_path/"collected"
    output=tmp_path/"report"
    shape=(campaign.nmechanical("KL"),*campaign.GRID)
    electric=(campaign.electric_order("KL")+1,*campaign.GRID)
    for case in campaign.all_cases():
        if not campaign.included("pilot",case):
            continue
        key=campaign.case_id(case)
        _,_,_,_,passport_digest=campaign.physical_data(case[0],case[1],case[2])
        folder=root/key
        folder.mkdir(parents=True)
        is_mvi=case[3]=="MVI"
        rank=case[4]
        sha=("b"*40 if is_mvi and rank==8 else "a"*40)
        quality=(True if not is_mvi or rank==3 else False)
        record={
            "id":key,"status":"COMPLETED_RESEARCH_ONLY",
            "physical_passport_sha256":passport_digest,
            "git_sha":sha,
            "full_nodal_equilibrium_satisfied":quality,
            "full_nodal_norm":1.e-10 if quality else 1.e-3,
            "solution_status":"CONVERGED_PROJECTED" if not quality else
                              "MVI_FIXED_RANK_FULL_NODAL_STATIONARY",
        }
        (folder/"EVIDENCE.json").write_text(json.dumps(record))
        np.savez_compressed(
            folder/"FULL_FIELDS.npz",
            mechanical=np.ones(shape),electrical=np.ones(electric),
            radial_outer=np.ones(campaign.GRID))
    # This report must raise because stale source ranks cannot be compared.
    with pytest.raises(RuntimeError,match="INCOMPLETE_NATIVE_FDM_MVI_GRADIENT_CAMPAIGN"):
        campaign.compare(root,output,"pilot")
    rows=json.loads((output/"COMPARISON.json").read_text())["results"]
    assert len(rows)==12
    assert {r["status"] for r in rows}=={
        "COMPARED_RESEARCH_ONLY","SOLVER_SOURCE_REVISION_MISMATCH"}
    full=[r for r in rows if r["comparison_quality"]=="FULL_NODAL_RESEARCH_ONLY"]
    projected=[r for r in rows if r["comparison_quality"]=="PROJECTED_ONLY_NOT_FULL_EQUILIBRIUM"]
    stale=[r for r in rows if r["status"]=="SOLVER_SOURCE_REVISION_MISMATCH"]
    assert len(full)==4 and len(projected)==4 and len(stale)==4
    report=json.loads((output/"COMPARISON.json").read_text())
    assert report["full_nodal_compared_pairs"]==4
    assert report["projected_only_compared_pairs"]==4
    assert report["compared_pairs"]==8
    assert report["status"]=="INCOMPLETE"


def test_comparison_rejects_matching_unknown_revisions(tmp_path):
    root=tmp_path/"collected"
    for case in campaign.all_cases():
        if not campaign.included("pilot",case):
            continue
        folder=root/campaign.case_id(case)
        folder.mkdir(parents=True)
        digest=campaign.physical_data(case[0],case[1],case[2])[-1]
        record=dict(id=campaign.case_id(case),status="COMPLETED_RESEARCH_ONLY",
                    physical_passport_sha256=digest,git_sha="UNKNOWN")
        (folder/"EVIDENCE.json").write_text(json.dumps(record))
    output=tmp_path/"report"
    with pytest.raises(RuntimeError,match="INCOMPLETE_NATIVE_FDM_MVI_GRADIENT_CAMPAIGN"):
        campaign.compare(root,output,"pilot")
    report=json.loads((output/"COMPARISON.json").read_text())
    assert report["compared_pairs"]==0
    assert all(row["status"]=="SOLVER_SOURCE_REVISION_UNVERIFIED"
               for row in report["results"])
