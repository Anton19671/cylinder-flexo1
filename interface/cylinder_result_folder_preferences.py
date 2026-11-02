"""Persist the selected result root separately from physical CASE files."""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path

def _preference_file():
    root=os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CONFIG_HOME")
    return (Path(root) if root else Path.home()/".config")/"CylinderFlexo1"/"results_folder.json"

def load_results_folder():
    try:
        data=json.loads(_preference_file().read_text(encoding="utf-8"))
        folder=data.get("last_folder") if isinstance(data,dict) else None
        return folder if isinstance(folder,str) and folder.strip() else None
    except (OSError,UnicodeError,ValueError):
        return None

def save_results_folder(folder):
    folder=Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("Выбранная папка результатов не существует")
    path=_preference_file()
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=str(path.parent),suffix=".tmp")
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as out:
            json.dump({"last_folder":str(folder)},out,ensure_ascii=False,indent=2)
            out.write("\n")
        os.replace(temp,path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return str(folder)
