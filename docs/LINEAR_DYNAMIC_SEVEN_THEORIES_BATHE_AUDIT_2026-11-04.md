# Linear dynamic verification: seven shell theories (4 November 2026)

## Scope and provenance

Repository `Anton19671/cylinder-flexo1`, branch `main`, commit
`0995b0067ed267d5b4d2024d53de68ba05ba2589`.
Source: GitHub Actions run
[41800966283](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41800966283);
its fourteen independently generated physical solver outputs and the archived
strict component-field audit (`bathe-seven-theories-current-run-full-field-audit`).

Seven models: KL, FSDT, Reddy, CUF TE1, CUF TE2, CUF TE3, CUF TE4.
Linear Hooke material, zero initial state, clamped axial edges, 25 Hz
harmonic radial dead pressure, grid 9x9, Bathe method with 0.002 s
one full step and physical stages at 0.001171572875 s and 0.002 s.
MVI fixed rank 1 versus independent full-nodal matrix-free FDM.

All 14 physical solver jobs completed successfully (including TE4 FDM).
**The independent field comparison FAILED**: 62 named component-field
checks exceeded the *existing*, unchanged nodewise acceptance limits.
This is not an all-model verification PASS.

## Full generalized mechanical state comparison

Percent values are 100 * norm(MVI - FDM) / norm(FDM), over every
generalized mechanical coefficient and every node of the 9x9 grid.
Entries below are from the final Bathe stage, t = 0.002 s.

| Theory | Displacement (%) | Velocity (%) | Acceleration (%) |
| --- | ---: | ---: | ---: |
| KL | 0.0000002433 | 0.0000007012 | 0.0000009330 |
| FSDT | 0.0003565381 | 0.0009887729 | 0.0012736807 |
| Reddy | 0.0000210484 | 0.0000582146 | 0.0000755303 |
| CUF TE1 | 0.0005157091 | 0.0013962273 | 0.0019109947 |
| CUF TE2 | 0.0008802159 | 0.0023982530 | 0.0033060828 |
| CUF TE3 | 0.0029663533 | 0.0076582016 | 0.0108457264 |
| CUF TE4 | 0.0052176062 | 0.0132476223 | 0.0196935697 |

Named component-field checks failing at GAMMA / END, respectively:
KL 0/0; FSDT 0/6; Reddy 0/2; CUF TE1 0/8;
CUF TE2 0/5; CUF TE3 3/12; CUF TE4 10/16.
Total 62.

## Physical displacement recovery for CUF TE4 — basis correction

**Correction (4 November 2026):** The original version of this audit
incorrectly described the active CUF TE basis as `z**k`. That power basis
belongs to the separate legacy/general provider
`models/final_te_basis_provider.py`, **not** the variational solver used
in the archived Bathe calculation.

The actual physical CUF TE1..TE4 solver uses
`flexoelectric/cuf_g3d_weak.py::te_basis`:

    A(x,s,z) = sum_{k=0}^N A_k(x,s) * (2*z/h)**k

for each of A in {U,V,W}, with h = 0.05 m, -h/2 <= z <= h/2.
Thus the physical inner and outer face factors are `(-1)**k` and
`(+1)**k`, rather than `(-h/2)**k` and `(+h/2)**k`.

The previous surface-reconstruction numerical table was derived with
the wrong basis and **must not be used** for quantitative physical-field
comparisons. The radial mid-surface displacement, which is simply W_0
because all higher-order powers vanish at z=0, is unaffected. In the
run's 9x9 grid at (x-index=4, s-index=4), t=0.002 s, this gives
MVI = -2.877936954e-6 m and FDM = -2.877935860e-6 m.
That central value alone does not prove agreement over the shell surface.

A further recovery audit must obtain face fields using the normalized
basis above, and compare original measured NPZ histories, not this
withdrawn table. No physical solver calculations were made to perform
the correction.

## Independent FDM accuracy sensitivity

Comparing the earlier archived CUF TE4 FDM state from
[run 41645787449](https://github.com/Anton19671/cylinder-flexo1/actions/runs/41645787449)
against the tighter FDM in 41800966283:

- GAMMA: displacement norm changes by 0.0177267%.
- END: displacement norm changes by 0.00639744%, velocity by
  0.01498030%, acceleration by 0.06186675%.

The last result demonstrates sensitivity of the full-nodal numerical
reference to its Krylov precision settings. It does not show which
solution is more accurate independently of a tighter or direct oracle.
The current TE4 work budget is 1200 Jv actions, compared with 300
before the correction; the prescribed accuracy target was not relaxed.

## Remaining proof obligations

1. Do not classify the seven-model comparison as PASS: reproduce and
   explain the recorded nodewise discrepancies without altering the
   published limits or changing any constitutive/kinematic equation.
2. For TE3 and TE4, separately compare **reconstructed physical**
   U(z), V(z), W(z) at surfaces and mid-thickness, and individual
   Taylor coefficients; do not mix their units or numerical scales.
3. Check FDM precision sensitivity for each model with an independently
   validated numerical reference; distinguish solver precision from
   fixed-rank truncation.
4. Verify true time-step convergence over multiple accepted physical
   time steps, and spatial-grid convergence.
5. Examine MVI ranks 2 through 8 against the same physical baseline
   before claiming rank convergence for any model.

This document reports analysis of **existing** GitHub Actions archives.
No new physical calculation has been launched by this report.
