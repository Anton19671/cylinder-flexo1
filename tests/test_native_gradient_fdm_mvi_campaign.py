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
            with np.load(folder/"loads.npz",allow_pickle=False) as z:
                np.testing.assert_array_equal(z["mechanical_load"],forcing)
    assert len(hashes)==1


@pytest.mark.parametrize("rank",range(1,9))
def test_rank_keeps_every_fourier_partner_not_replaced(rank):
    partner=campaign.fourier_seed(rank,campaign.GRID[1])
    assert partner.shape==(rank,campaign.GRID[1])
    assert np.linalg.matrix_rank(partner,tol=1.e-12)==rank
