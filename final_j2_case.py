# -*- coding: utf-8 -*-
"""Legacy import/CLI shim. Implementation lives in a topic directory."""
from pathlib import Path as _CylinderPath
_cylinder_root = _CylinderPath(__file__).resolve().parent
_cylinder_name = _CylinderPath(__file__).name
_cylinder_matches = [p for p in _cylinder_root.iterdir() if p.is_dir() and (p / _cylinder_name).is_file()]
if len(_cylinder_matches) != 1:
    raise ImportError("Expected exactly one relocated cylinder module: " + _cylinder_name)
_cylinder_source = _cylinder_matches[0] / _cylinder_name
exec(compile(_cylinder_source.read_bytes(), str(_cylinder_source), "exec"), globals())
