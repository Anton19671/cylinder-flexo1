# -*- coding: utf-8 -*-
"""Public entrypoint; real implementation lives in its thematic package."""
from pathlib import Path as _Path
from cylinder_paths import activate as _activate
_root = _activate()
_real = _root / "interface" / "FINAL_GUI.py"
if not _real.is_file():
    raise ImportError("Missing relocated entrypoint " + _Path(__file__).name)
__file__ = str(_real)
exec(compile(_real.read_bytes(), str(_real), "exec"), globals())
