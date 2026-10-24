#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z2 static-hygiene guard using only the Python standard library.

The original audit used pyflakes.  The release itself must not depend on a
third-party linter, so this check enforces the two high-value invariants that
can be checked deterministically with ast/symtable:
  * no unused top-level imports in production modules;
  * no obviously dead local stores under the same conservative scope model;
  * no referenced names lacking a binding in their Python scope.
It also guards the audit's special FULL-GL+Birger import: ``fg`` is retained
because it is actually used by the exact analytic tangent assembly, and it
checks the explicit ``prev_R`` binding required by reviewer item V-10.
"""
from __future__ import annotations
import ast
import builtins
import symtable
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def _is_check_script(name):
    return "_SELF_CHECK" in name or name.startswith(("RUN_", "FINAL_RELEASE_REGRESSION"))


def _production_files():
    for p in sorted(ROOT.glob("*.py")):
        if _is_check_script(p.name):
            continue
        yield p




def _unused_top_imports(tree):
    refs={n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
    out=[]
    for node in tree.body:
        if isinstance(node,ast.Import):
            for a in node.names:
                name=a.asname or a.name.split('.')[0]
                if name not in refs:
                    out.append((node.lineno,name))
        elif isinstance(node,ast.ImportFrom) and node.module!='__future__':
            for a in node.names:
                if a.name=='*':
                    continue
                name=a.asname or a.name
                if name not in refs:
                    out.append((node.lineno,name))
    return out


def _unresolved_names(source, filename):
    """Return referenced names without a binding in their Python scope."""
    root = symtable.symtable(source, filename, "exec")
    module_bound = {
        symbol.get_name()
        for symbol in root.get_symbols()
        if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()
    }
    star_import = "import *" in source
    dunder = {
        "__file__", "__name__", "__doc__", "__spec__", "__loader__",
        "__package__", "__builtins__", "__path__",
    }
    out = []

    def walk(table):
        for symbol in table.get_symbols():
            name = symbol.get_name()
            if (
                not symbol.is_referenced()
                or symbol.is_parameter()
                or symbol.is_imported()
                or symbol.is_free()
            ):
                continue
            if symbol.is_declared_global():
                if name not in module_bound and not hasattr(builtins, name):
                    out.append((table.get_name(), table.get_lineno(), name))
                continue
            if (
                (symbol.is_global() or table.get_type() == "module")
                and name not in module_bound
                and not hasattr(builtins, name)
                and name not in dunder
                and not star_import
            ):
                out.append((table.get_name(), table.get_lineno(), name))
        for child in table.get_children():
            walk(child)

    walk(root)
    return out




def _unused_local_stores(tree):
    out=[]
    for fn in ast.walk(tree):
        if not isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):
            continue
        stores={}
        class StoreVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self,node):
                if node is fn:
                    for st in node.body:
                        self.visit(st)
            visit_AsyncFunctionDef=visit_FunctionDef
            def visit_Lambda(self,node):
                return
            def visit_ClassDef(self,node):
                return
            def visit_Name(self,node):
                if isinstance(node.ctx,ast.Store):
                    stores.setdefault(node.id,node.lineno)
        StoreVisitor().visit(fn)
        # Descendant loads count, because a nested function may close over an
        # outer local.  Descendant stores do not count as outer stores above.
        loads={n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
        for name,line in stores.items():
            if name not in loads and not name.startswith('_'):
                out.append((fn.name,line,name))
    return out


def run_checks(verbose=True):
    import_issues=[]; local_issues=[]; name_issues=[]
    for p in _production_files():
        source=p.read_text(encoding='utf-8')
        tree=ast.parse(source,filename=str(p))
        for line,name in _unused_top_imports(tree):
            import_issues.append((p.name,line,name))
        for fn,line,name in _unused_local_stores(tree):
            local_issues.append((p.name,fn,line,name))
        for scope,line,name in _unresolved_names(source,str(p)):
            name_issues.append((p.name,scope,line,name))
    if import_issues:
        raise AssertionError(f'Z2 unused production imports: {import_issues}')
    if local_issues:
        raise AssertionError(f'Z2 obvious dead production locals: {local_issues}')
    if name_issues:
        raise AssertionError(f'Z2 unresolved production scope names: {name_issues}')

    p=ROOT/'cylinder_cuf_stage10d_strong_fdm_full_gl_birger.py'
    txt=p.read_text(encoding='utf-8')
    if 'import cylinder_cuf_stage10b_strong_fdm_full_gl_elastic as fg' not in txt:
        raise AssertionError('Z2: FULL GL + Birger fg tangent dependency was removed')
    if 'fg._assemble_exact_tangent_with_end_bcs' not in txt:
        raise AssertionError('Z2: fg import is not connected to analytic tangent assembly')
    required_prev_r=('prev_R=R0','base_residual=prev_R','prev_R=Rnew')
    missing=[token for token in required_prev_r if token not in txt]
    if missing:
        raise AssertionError(f'Z2: reviewer V-10 explicit prev_R binding is incomplete: {missing}')
    if 'base_R=R0 if k==1 else Rnew' in txt:
        raise AssertionError('Z2: reviewer V-10 legacy implicit Rnew binding returned')

    report={
        'status':'PASS',
        'production_files_checked':sum(1 for _ in _production_files()),
        'unused_top_level_imports':0,
        'obvious_dead_local_stores':0,
        'unresolved_scope_names':0,
        'full_gl_birger_fg_dependency':'USED',
        'full_gl_birger_prev_R_binding':'EXPLICIT',
    }
    if verbose:
        print('Z2_SELF_CHECK: PASS')
        print(f"  production modules checked: {report['production_files_checked']}")
        print('  unused top-level imports: 0')
        print('  obvious dead local stores: 0')
        print('  unresolved scope names: 0')
        print('  FULL GL + Birger fg dependency: USED, retained')
        print('  FULL GL + Birger prev_R binding: EXPLICIT')
    return report

if __name__=='__main__':
    run_checks(True)
