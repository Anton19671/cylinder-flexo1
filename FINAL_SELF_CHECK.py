#!/usr/bin/env python3
"""Run only coupled elastic/flexoelectric G3D and Gz regressions."""
from cylinder_paths import activate
def main():
    root=activate()
    try:import pytest
    except ImportError as exc:
        raise SystemExit("Install pytest before running the self checks") from exc
    raise SystemExit(pytest.main(["-q",str(root/"tests")]))
if __name__=="__main__":main()
