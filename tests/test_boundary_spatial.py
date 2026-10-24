#!/usr/bin/env python3
"""Constrained spatial first variations, MVI half-steps and timed end data."""
import copy
import unittest
import numpy as np

from cylinder_boundary_spatial import (
    BoundarySpatialProblem, affine_trace_space, separated_profile_directions,
)
from cylinder_functional_assembly import CylinderFunctional, surface_trapezoid
from cylinder_variational_boundary import combine_functional_and_boundary
from cylinder_boundary_editor import merge_boundary_controls
from flexo_gradient_g3d import G3DMaterial


class BoundarySpatialTest(unittest.TestCase):
    def setUp(self):
        self.ops=surface_trapezoid(4,4,1.2,2.)
        self.rng=np.random.default_rng(48)

    def functional(self,theory="CUF_TE1",electric=False):
        mat=G3DMaterial(100.,.25,.08,.01 if electric else 0.,
                        -.02 if electric else 0.,1.5 if electric else 0.)
        return CylinderFunctional(theory,mat,radius_m=2.,thickness_m=.2,
                                  surface_operators=self.ops,n_gauss=2)

    def test_all_theories_impose_end_traces_and_electrodes(self):
        for theory in ("KL","FSDT","REDDY","CUF_TE1","CUF_TE2","CUF_TE3","CUF_TE4"):
            for electric in (False,True):
                with self.subTest(theory=theory,electric=electric):
                    f=self.functional(theory,electric)
                    q=np.zeros((f.n_mechanical,*self.ops.shape))
                    p=np.zeros((2,*self.ops.shape)) if electric else None
                    spec={"mechanical":{"left":{"support":"CLAMPED","higher":"FIX_FIRST_NORMAL"}},
                          "electrical":({"outer":{"mode":"POTENTIAL","value":2.},
                                         "inner":{"mode":"POTENTIAL","value":-2.}} if electric else {})}
                    problem=BoundarySpatialProblem(f,bc_spec=spec)
                    a=problem.assemble(q,p)
                    np.testing.assert_allclose(a.space.matrix@a.space.basis,0,atol=3e-14)
                    np.testing.assert_allclose(a.space.defect(a.space.origin),0,atol=3e-14)
                    feasible=a.space.origin+self.rng.normal(0,1e-4,a.space.origin.shape)@(
                        a.space.basis@a.space.basis.T)
                    np.testing.assert_allclose(a.space.defect(feasible),0,atol=3e-14)
                    if electric:
                        field=a.space.origin[q.size:].reshape(p.shape)
                        np.testing.assert_allclose(field[0]+field[1],2,atol=3e-14)
                        np.testing.assert_allclose(field[0]-field[1],-2,atol=3e-14)
                        self.assertFalse(a.edge.gauge_required)

    def test_constrained_energy_gradient_includes_bulk_and_edge_work(self):
        f=self.functional("FSDT",True)
        q=np.zeros((5,*self.ops.shape));p=np.zeros((2,*self.ops.shape))
        spec={"mechanical":{"left":{"support":"CLAMPED"},
              "right":{"spring":[.5]*5,"traction":[.01]*5,"double_traction":[.001]*5}},
              "electrical":{"outer":{"mode":"POTENTIAL","value":.02}}}
        problem=BoundarySpatialProblem(f,bc_spec=spec)
        initial=problem.assemble(q,p)
        u=initial.space.origin+initial.space.basis@self.rng.normal(0,1e-4,initial.space.basis.shape[1])
        q=u[:q.size].reshape(q.shape);p=u[q.size:].reshape(p.shape)
        a=problem.assemble(q,p)
        direction=a.space.basis@self.rng.normal(0,1e-4,a.space.basis.shape[1])
        eps=1e-3
        def energy(sign):
            v=u+sign*eps*direction
            return problem.assemble(v[:q.size].reshape(q.shape),v[q.size:].reshape(p.shape)).potential_j
        np.testing.assert_allclose((energy(1)-energy(-1))/(2*eps),
                                   a.full_residual@direction,rtol=2e-5,atol=1e-8)
        self.assertLess(a.constraint_max_defect,1e-12)

    def test_insulated_gauge_and_nanocoulomb_balance(self):
        f=self.functional(electric=True)
        q=np.zeros((f.n_mechanical,*self.ops.shape));p=np.zeros((2,*self.ops.shape))
        a=BoundarySpatialProblem(f).assemble(q,p)
        self.assertEqual(a.space.labels,("electric volume-mean gauge",))
        self.assertTrue(a.edge.gauge_required)
        charge=np.zeros_like(p);charge[0]=1e-12
        # Both bulk and end charges are individually unbalanced, but their
        # physical total is zero. Compatibility must be checked only once.
        total=float(np.sum(f.W*charge[0]))
        end_value=-total/(f.h*sum(self.ops.ws))
        spec={"electrical":{"right":{"mode":"CHARGE","value":end_value}}}
        combine_functional_and_boundary(f,q,p,free_charge=charge,bc_spec=spec)
        with self.assertRaisesRegex(ValueError,"zero total"):
            combine_functional_and_boundary(f,q,p,free_charge=charge)
        # A voltage electrode supplies the missing charge; no gauge equation.
        spec={"electrical":{"outer":{"mode":"POTENTIAL","value":0.}}}
        a=BoundarySpatialProblem(f,bc_spec=spec).assemble(q,p,free_charge=charge)
        self.assertFalse(a.edge.gauge_required)
        self.assertNotIn("electric volume-mean gauge",a.space.labels)

    def test_mvi_one_through_eight_frozen_partner_spaces(self):
        for rank in range(1,9):
            x=self.rng.normal(size=(1,rank,9))
            y=self.rng.normal(size=(1,rank,10))
            base=np.einsum("crx,crs->cxs",x,y).ravel()
            c=np.zeros((10,base.size))
            c[np.arange(10),np.arange(10)]=1.
            # Existing nonzero end lift is preserved in every half-step.
            for direction in ("Y","X"):
                with self.subTest(rank=rank,direction=direction):
                    t=separated_profile_directions(x,y,direction)
                    a=affine_trace_space(c,c@base,base,directions=t)
                    np.testing.assert_allclose(c@a.basis,0,atol=1e-13)
                    np.testing.assert_allclose(c@a.origin,c@base,atol=1e-13)
                    delta=a.basis@self.rng.normal(size=a.basis.shape[1])
                    # Homogeneous correction stays inside frozen-partner span.
                    reconstructed=t@np.linalg.solve(t.T@t,t.T@delta)
                    np.testing.assert_allclose(delta,reconstructed,atol=2e-12)

    def test_no_full_grid_fallback_for_unreachable_mvi_trace(self):
        c=np.array([[1.,0.,0.]])
        with self.assertRaisesRegex(ValueError,"unreachable"):
            affine_trace_space(c,np.array([1.]),np.zeros(3),directions=np.array([[0.],[1.],[0.]]))
        with self.assertRaisesRegex(ValueError,"inconsistent"):
            affine_trace_space(np.array([[1.,0.],[2.,0.]]),np.array([1.,3.]),np.zeros(2))

    def test_physical_mvi_halfstep_uses_bulk_edge_and_electric_constraints(self):
        f=self.functional(electric=True)
        nc=f.n_mechanical+2
        x=self.rng.normal(0,1e-4,(nc,1,4));x[:,:,0]=0.
        x[-1]=-x[-2]
        y=np.ones((nc,1,4))
        fields=np.einsum("crx,crs->cxs",x,y)
        q,p=fields[:f.n_mechanical],fields[f.n_mechanical:]
        spec={"mechanical":{"left":{"support":"CLAMPED"}},
              "electrical":{"outer":{"mode":"POTENTIAL","value":0.}}}
        problem=BoundarySpatialProblem(f,bc_spec=spec)
        for direction in ("Y","X"):
            with self.subTest(direction=direction):
                t=separated_profile_directions(x,y,direction)
                a=problem.assemble(q,p,method="MVI",directions=t)
                self.assertLess(a.constraint_max_defect,1e-12)
                delta=a.space.basis@self.rng.normal(0,1e-4,a.space.basis.shape[1])
                np.testing.assert_allclose(a.space.matrix@delta,0,atol=1e-15)
                eps=1e-3
                energies=[]
                for sign in (1,-1):
                    v=fields.ravel()+sign*eps*delta
                    qm=v[:q.size].reshape(q.shape);pm=v[q.size:].reshape(p.shape)
                    energies.append(combine_functional_and_boundary(f,qm,pm,bc_spec=spec)[0])
                np.testing.assert_allclose((energies[0]-energies[1])/(2*eps),
                                           a.full_residual@delta,rtol=3e-5,atol=1e-8)

    def test_projected_newton_equations_keep_reactions(self):
        c=np.array([[1.,0.,0.],[0.,1.,1.]])
        a=affine_trace_space(c,np.array([.2,0.]),np.zeros(3))
        k=np.array([[4.,1.,0.],[1.,3.,.4],[0.,.4,2.]])
        force=np.array([0.,1.,-1.])
        u=a.origin.copy()
        correction=np.linalg.solve(a.tangent(lambda v:k@v),-a.residual(k@u-force))
        u+=a.basis@correction
        np.testing.assert_allclose(a.defect(u),0,atol=1e-14)
        np.testing.assert_allclose(a.residual(k@u-force),0,atol=1e-14)
        self.assertGreater(np.linalg.norm(k@u-force),.1)  # support reactions

    def test_dynamic_boundary_and_inertia_use_same_spatial_constraints(self):
        f=self.functional("CUF_TE1",True)
        q=np.zeros((f.n_mechanical,*self.ops.shape));p=np.zeros((2,*self.ops.shape))
        law={"kind":"RAMP","duration_s":1.,"amplitude":1.}
        spec={"mechanical":{"left":{"support":"CLAMPED","prescribed":[.02]*f.n_mechanical,"time_law":law}},
              "electrical":{"outer":{"mode":"POTENTIAL","value":2.,"time_law":law}}}
        problem=BoundarySpatialProblem(f,bc_spec=spec,calculation_kind="DYNAMIC")
        a=problem.assemble(q,p,time_s=.25)
        b=problem.assemble(q,p,time_s=.5)
        np.testing.assert_allclose(b.space.target,2*a.space.target)
        accel=np.ones_like(q)*.01
        inertia=problem.assemble(q,p,time_s=.25,acceleration=accel,density_kg_m3=1300.)
        expected=np.concatenate((f.mass_action(accel,density_kg_m3=1300.).ravel(),np.zeros(p.size)))
        np.testing.assert_allclose(inertia.full_residual-a.full_residual,expected)
        with self.assertRaisesRegex(ValueError,"time_s"):
            problem.assemble(q,p)
        with self.assertRaisesRegex(ValueError,"MVI requires"):
            problem.assemble(q,p,method="MVI",time_s=.25)

    def test_gui_apply_saves_changed_controls_without_manual_sync(self):
        m={"left":{"support":"FREE","higher":"NATURAL"}}
        e={"outer":{"mode":"INSULATED","value":""}}
        previous=copy.deepcopy((m,e))
        data={"mechanical":{"left":{"support":"CUSTOM","fixed_components":[0],"spring":[1,0,0]}},
              "electrical":{"outer":{"mode":"POTENTIAL","value":3.}}}
        # No changed controls: direct JSON is authoritative.
        self.assertEqual(merge_boundary_controls(data,m,e,previous),data)
        m["left"]["higher"]="FIX_FIRST_NORMAL"
        e["outer"]={"mode":"CHARGE","value":".002"}
        merged=merge_boundary_controls(data,m,e,previous)
        self.assertEqual(merged["mechanical"]["left"]["support"],"CUSTOM")
        self.assertEqual(merged["mechanical"]["left"]["higher"],"FIX_FIRST_NORMAL")
        self.assertEqual(merged["electrical"]["outer"],{"mode":"CHARGE","value":.002})
        self.assertEqual(data["electrical"]["outer"]["value"],3.)


if __name__=="__main__":
    unittest.main()
