# Native stop/resume and projected static completion — 18 November 2026

Base: `892a389155258bef9ca84747b31abdd0dc87800c`.

Corrected four related execution defects:

- Native MVI restart rejected the `STOPPED_BY_USER` status written by its
  GUI. Explicit `--resume` now accepts this status subject to the existing
  input, implementation, rank, state fingerprint and history checks.
- Old STOP/PAUSE marker files immediately stopped or blocked the resumed
  process. They are removed only after checkpoint validation under the
  exclusive output lock. Rejected restarts retain markers and prior summary.
- Cooperative user cancellation was recorded as FAILED. A dedicated
  InterruptedError subclass now identifies user cancellation in both native
  FDM runners and MVI. MVI also retains the cause when a reduced driver
  converts the interruption into a CANCELLED return value. Ordinary I/O
  interruptions remain failures. No partial step becomes accepted.
- The public CLI returned failure for an explicitly accepted projected
  stationary MVI result. It now recognizes that completed run, requiring
  the projected status, static kind, exactly one accepted step and explicit
  absence of full nodal equilibrium. Scientific certification remains false.

Validation: 131 tests and 4 subtests passed. Tests cover actual marker-file
handling, mocked interrupted/restarted time drivers, preserved rejected
checkpoints, both FDM failure summaries, public CLI exit codes and repository
import integrity. One external pytest/anyio deprecation warning.

No physical cylinder calculation or physical time history was launched.
Commit uses `[skip ci]` to avoid automatic physical workflow triggers.
Physical convergence and higher-order boundary closure remain outside the
evidence established by these execution fixes.
