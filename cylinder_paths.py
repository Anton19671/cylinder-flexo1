# -*- coding: utf-8 -*-
"""Import roots for the coupled elastic cylinder solver only."""
from pathlib import Path
import sys
SOURCE_FOLDERS=("interface","mechanics","flexoelectric","physics","numerics","tests")
def activate():
    root=Path(__file__).resolve().parent
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    at=sys.path.index(str(root))+1
    for folder in SOURCE_FOLDERS:
        p=root/folder
        if not p.is_dir():raise FileNotFoundError(f"Missing source directory: {p}")
        if str(p) not in sys.path:sys.path.insert(at,str(p));at+=1
    return root
__all__=("SOURCE_FOLDERS","activate")
