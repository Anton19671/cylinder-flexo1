# -*- coding: utf-8 -*-
"""Import setup for the organized cylinder-flexo1 repository.

Internal module names remain unchanged. The root contains ONLY public
entrypoints; implementations are in named source directories.
"""
from pathlib import Path
import sys

SOURCE_FOLDERS = (
    "interface", "models", "mechanics", "mvi", "mkr",
    "flexoelectric", "physics", "numerics", "results", "legacy", "tests",
)

def activate():
    root = Path(__file__).resolve().parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    insert_at = sys.path.index(root_str) + 1
    for folder in SOURCE_FOLDERS:
        path = root / folder
        if not path.is_dir():
            raise FileNotFoundError("Missing cylinder-flexo1 folder: " + str(path))
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(insert_at, path_str)
            insert_at += 1
    return root

__all__ = ("SOURCE_FOLDERS", "activate")
