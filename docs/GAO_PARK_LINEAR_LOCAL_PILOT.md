# Isolated linear local-kernel pilot

This branch does not modify production solvers or the running baseline
campaign. The Gao-Park benchmark is strictly LINEAR, mechanical CUF.
The unchanged local first variation is linear in the six mechanical
surface jets. With dimensionally scaled coordinates x, its Jacobian K is
constant and the elastic energy is 1/2 x^T K x. Construct K directly from
all independent unit-coordinate calls to the original Gauss-integrated
local kernel. Check zero state, symmetry and twelve independently sampled
full-jet states against the direct kernel before enabling it. Reject an
invalid or inaccurate local cache. Preserve the original native FDM and
fixed-rank MVI spatial operators, boundary equations, iterations, ranks
and convergence criteria. No substitution of a full-grid solution for MVI.

Pilot: TE1 and TE4, ell=0.5 micrometre, grid 21x9, FDM plus MVI ranks 1–8;
each result is uploaded independently. These are additional verification
runs, not evidence of completed grid/rank convergence. Expand only after
local equality and actual solver results have been checked.
