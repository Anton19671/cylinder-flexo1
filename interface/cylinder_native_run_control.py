"""GUI-owned cooperative controls for research CASE subprocesses.

The solver checks PAUSE and STOP files BETWEEN numerical steps.
Control files never change a physical equation, integrator, rank or
method. Stop inside an active nonlinear iteration is handled by GUI
process termination; only accepted completed steps are committed.
"""
from __future__ import annotations
import time
from pathlib import Path


def cooperate(output_dir):
    directory=Path(output_dir)
    stop=directory/"CONTROL_STOP"
    pause=directory/"CONTROL_PAUSE"
    while True:
        if stop.exists():
            raise InterruptedError("NATIVE_USER_STOP_REQUESTED")
        if not pause.exists():
            return
        time.sleep(.2)


__all__=("cooperate",)
