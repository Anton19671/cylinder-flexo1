from __future__ import annotations

import os
import runpy
import sys
import traceback
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
LOG = BASE / "GUI_RUNTIME_LOG.txt"
os.chdir(BASE)

with LOG.open("a", encoding="utf-8", buffering=1) as stream:
    stream.write("\n" + "=" * 72 + "\n")
    stream.write(f"GUI START: {datetime.now().isoformat(timespec='seconds')}\n")
    stream.write(f"PYTHON: {sys.executable}\n")
    stream.write(f"FOLDER: {BASE}\n")
    stream.flush()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = stream
    sys.stderr = stream
    try:
        runpy.run_path(str(BASE / "FINAL_GUI.py"), run_name="__main__")
    except BaseException:
        traceback.print_exc(file=stream)
        stream.flush()
        raise
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
