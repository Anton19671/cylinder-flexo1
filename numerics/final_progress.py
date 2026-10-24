"""Lightweight optional progress/event channel for GUI monitoring.

This module is deliberately independent of solver mechanics.  Calls are no-ops
unless ``configure(output_dir, ...)`` has been called by the top-level driver.
Events are written atomically to LIVE_PROGRESS.json and appended to
LIVE_PROGRESS.jsonl.  Any I/O error is swallowed so diagnostics can never alter
calculation results.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_writer = None
_context = {}


def _jsonable(v: Any):
    try:
        import numpy as np
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (np.bool_,)): return bool(v)
        if isinstance(v, np.ndarray):
            if v.ndim == 0: return _jsonable(v.item())
            return None
    except Exception:
        pass
    if isinstance(v, (str, int, float, bool)) or v is None: return v
    if isinstance(v, Path): return str(v)
    if isinstance(v, dict): return {str(k): _jsonable(x) for k, x in v.items() if _jsonable(x) is not None}
    if isinstance(v, (list, tuple)): return [_jsonable(x) for x in v if _jsonable(x) is not None]
    return str(v)


class ProgressWriter:
    def __init__(self, output_dir, **context):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.live_path = self.output_dir / "LIVE_PROGRESS.json"
        self.log_path = self.output_dir / "LIVE_PROGRESS.jsonl"
        self.start = time.monotonic()
        self.seq = 0
        self.context = dict(context)

    def set_context(self, **kw):
        self.context.update({k: v for k, v in kw.items() if v is not None})

    def emit(self, event: str, **data):
        self.seq += 1
        row = {
            "seq": self.seq,
            "event": str(event),
            "elapsed_s": time.monotonic() - self.start,
            "time_unix": time.time(),
            **self.context,
            **data,
        }
        row = _jsonable(row)
        try:
            tmp = self.live_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.live_path)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:
            pass
        return row


def configure(output_dir, **context):
    global _writer
    _writer = ProgressWriter(output_dir, **context)
    return _writer


def reset():
    global _writer, _context
    _writer = None
    _context = {}


def set_context(**kw):
    global _context
    _context.update({k: v for k, v in kw.items() if v is not None})
    if _writer is not None:
        _writer.set_context(**kw)


def emit(event: str, **data):
    if _writer is None:
        return None
    merged = dict(_context)
    merged.update(data)
    return _writer.emit(event, **merged)


def is_configured():
    return _writer is not None
