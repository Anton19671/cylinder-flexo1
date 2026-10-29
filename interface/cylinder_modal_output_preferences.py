"""One durable GUI preference for the latest modal output folders.

No CASE is written while a folder is being selected. Preference data is kept
separately from immutable, reproducible input CASE files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _preference_file() -> Path:
    root = (os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CONFIG_HOME'))
    return (Path(root) if root else Path.home() / '.config') / 'CylinderFlexo1' / 'modal_folders.json'


def _key(case_path) -> str | None:
    return (str(Path(case_path).expanduser().resolve()).casefold()
            if case_path is not None else None)


def _load_data() -> dict:
    try:
        raw = json.loads(_preference_file().read_text(encoding='utf-8'))
    except (FileNotFoundError, OSError, UnicodeError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def load_modal_folder(case_path=None) -> str | None:
    """Restore per-CASE choice or, for unsaved drafts, last selected folder."""
    data = _load_data()
    if case_path is None:
        value = data.get('last_folder')
    else:
        choices = data.get('by_case', {})
        value = choices.get(_key(case_path), data.get('last_folder')) if isinstance(choices, dict) else data.get('last_folder')
    return value if isinstance(value, str) and value.strip() else None


def save_modal_folder(folder: str, case_path=None) -> bool:
    """Overwrite a single preference file only when the selection changes."""
    folder = str(folder).strip()
    if not folder:
        return False
    data = _load_data()
    choices = data.get('by_case', {})
    if not isinstance(choices, dict):
        choices = {}
    key = _key(case_path)
    prior = choices.get(key) if key is not None else data.get('last_folder')
    if prior == folder and data.get('last_folder') == folder:
        return False
    data['format'] = 'CYLINDER_MODAL_FOLDER_PREFERENCES_V1'
    data['last_folder'] = folder
    if key is not None:
        choices[key] = folder
    data['by_case'] = choices
    path = _preference_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def save_folder_in_modal_case(case_path, folder: str) -> bool:
    """Save only output_dir in an existing MODAL CASE, with atomic replacement.

    Selecting an output directory is an explicit edit. Persist it in the
    authoritative case so the main Run button cannot silently use an old path.
    No AUTO CASE is created here and physical settings remain untouched.
    """
    if case_path is None:
        return False
    source = Path(case_path).expanduser().resolve()
    if not source.is_file():
        return False
    from cylinder_case_io import load_case_file, save_case_file
    case = load_case_file(source)
    if (case.get('solver_backend') != 'VARIATIONAL_REFERENCE'
            or str(case.get('calculation_kind', '')).upper() != 'MODAL'):
        return False
    folder = str(folder).strip()
    if not folder:
        raise ValueError('Modal result folder must not be empty')
    if str(case.get('output_dir', '')).strip() == folder:
        return False
    case['output_dir'] = folder
    # A temporary file with the same suffix supports both JSON and YAML;
    # the original file is never truncated before a full replacement exists.
    import tempfile
    fd, temp = tempfile.mkstemp(
        dir=str(source.parent), prefix=source.stem + '.folder-',
        suffix=source.suffix)
    os.close(fd)
    temporary = Path(temp)
    try:
        save_case_file(temporary, case)
        os.replace(temporary, source)
    finally:
        temporary.unlink(missing_ok=True)
    return True


__all__ = ('load_modal_folder', 'save_modal_folder', 'save_folder_in_modal_case')
