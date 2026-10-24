# -*- coding: utf-8 -*-
"""Public entrypoint; real implementation lives in its thematic package."""
from pathlib import Path as _Path
from cylinder_paths import activate as _activate
_root = _activate()
_dest = next(
    (p for p in _root.iterdir() if p.is_dir() and (p / _Path(__file__).name).is_file()),
    None
)
if _dest is None:
    raise ImportError("Missing relocated entrypoint " + _Path(__file__).name)
_real = _dest / _Path(__file__).name
__file__ = str(_real)
exec(compile(_real.read_bytes(), str(_real), "exec"), globals())
