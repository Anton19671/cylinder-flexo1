#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build explicit Birger material diagram Phi(epsilon_i) from GUI data.

No fitted curve is introduced automatically.  Interpolation/extrapolation is
selected explicitly by the user.
"""
from __future__ import annotations
import ast, csv, math
from pathlib import Path
import numpy as np

CURVE_MODES=("TABLE","CSV","FORMULA","PIECEWISE","EXTERNAL_PYTHON")
EXTRAPOLATIONS=("ERROR","LINEAR","HOLD")

class BirgerCurveError(ValueError): pass

_ALLOWED_FUNCS={
    "sqrt":np.sqrt,"exp":np.exp,"log":np.log,"log10":np.log10,
    "sin":np.sin,"cos":np.cos,"tan":np.tan,"tanh":np.tanh,
    "abs":np.abs,"minimum":np.minimum,"maximum":np.maximum,
    "power":np.power,"where":np.where,
}
_ALLOWED_NAMES=set(_ALLOWED_FUNCS)|{"eps","pi","e"}
_ALLOWED_AST=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Call,ast.Name,ast.Load,
              ast.Constant,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.Mod,
              ast.USub,ast.UAdd,
              ast.Compare,ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE)


def _normalize_parameters(parameters):
    if parameters is None:
        return {}
    if not isinstance(parameters,dict):
        raise BirgerCurveError("formula parameters must be an object")
    out={}
    reserved=set(_ALLOWED_FUNCS)|{"eps","pi","e"}
    for raw_name,raw_value in parameters.items():
        name=str(raw_name).strip()
        if not name or not name.isidentifier():
            raise BirgerCurveError(f"invalid formula parameter name: {raw_name!r}")
        if name in reserved:
            raise BirgerCurveError(f"formula parameter name is reserved: {name}")
        try:value=float(raw_value)
        except Exception as exc:raise BirgerCurveError(f"formula parameter {name} must be numeric") from exc
        if not np.isfinite(value):raise BirgerCurveError(f"formula parameter {name} must be finite")
        out[name]=value
    return out


def _compile_formula(expr,parameters=None):
    text=str(expr).strip()
    if not text: raise BirgerCurveError("formula is empty")
    params=_normalize_parameters(parameters)
    allowed_names=_ALLOWED_NAMES|set(params)
    try: tree=ast.parse(text,mode="eval")
    except SyntaxError as exc: raise BirgerCurveError(f"invalid formula: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node,_ALLOWED_AST):
            raise BirgerCurveError(f"formula element is not allowed: {type(node).__name__}")
        if isinstance(node,ast.Name) and node.id not in allowed_names:
            raise BirgerCurveError(f"formula name is not allowed: {node.id}")
        if isinstance(node,ast.Call):
            if not isinstance(node.func,ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise BirgerCurveError("only approved mathematical functions may be called")
    return compile(tree,"<BirgerFormula>","eval"),params


def formula_callable(expr,parameters=None):
    code,params=_compile_formula(expr,parameters)
    def phi(eps):
        a=np.asarray(eps,float)
        env={"eps":a,"pi":math.pi,"e":math.e,**_ALLOWED_FUNCS,**params}
        with np.errstate(all="ignore"):
            y=np.asarray(eval(code,{"__builtins__":{}},env),float)
        if y.ndim==0: y=np.full_like(a,float(y))
        else: y=np.broadcast_to(y,a.shape).astype(float,copy=True)
        if not np.all(np.isfinite(y)) or np.any(y<0):
            raise BirgerCurveError("formula returned non-finite or negative stress intensity")
        return y
    phi.parameters=dict(params)
    return phi

def _clean_points(points):
    a=np.asarray(points,float)
    if a.ndim!=2 or a.shape[1]!=2 or a.shape[0]<2:
        raise BirgerCurveError("at least two [epsilon,sigma] points are required")
    if not np.all(np.isfinite(a)) or np.any(a[:,0]<0) or np.any(a[:,1]<0):
        raise BirgerCurveError("curve points must be finite and non-negative")
    order=np.argsort(a[:,0]); a=a[order]
    if np.any(np.diff(a[:,0])<=0): raise BirgerCurveError("epsilon values must be strictly increasing")
    # V10: the Birger intensity diagram may harden or form a plateau, but the
    # supplied branch must not descend.  No smoothing/correction is applied.
    dy=np.diff(a[:,1])
    tol=64.0*np.finfo(float).eps*max(1.0,float(np.max(np.abs(a[:,1]))))
    bad=np.flatnonzero(dy < -tol)
    if bad.size:
        i=int(bad[0])
        raise BirgerCurveError(
            "V10 Birger curve must be nondecreasing: "
            f"Phi({a[i+1,0]:.17g})={a[i+1,1]:.17g} < "
            f"Phi({a[i,0]:.17g})={a[i,1]:.17g}"
        )
    return a


def _csv_delimiter(sample):
    """Detect a two-column CSV delimiter without silently guessing data rows."""
    text=str(sample)
    try:
        dialect=csv.Sniffer().sniff(text,delimiters=",;\t")
        return dialect.delimiter
    except csv.Error:
        # Deterministic fallback for short files.  Semicolon/tab are preferred
        # because they also allow a decimal comma inside a numeric field.
        counts={";":text.count(";"),"\t":text.count("\t"),",":text.count(",")}
        delim=max(counts,key=counts.get)
        if counts[delim]<=0:
            raise BirgerCurveError(
                "cannot determine CSV delimiter; expected two columns separated by ',', ';' or TAB"
            )
        return delim


def _csv_number(text, *, decimal_comma=False):
    value=str(text).strip()
    if decimal_comma:
        value=value.replace(",",".")
    return float(value)


def read_csv_points(path):
    """Read a strict two-column Birger intensity diagram.

    Blank/comment lines are ignored.  At most one non-numeric header record is
    accepted, and it must precede every numeric data record.  Any malformed row
    after data have started is a hard error with its physical line number.
    """
    p=Path(path)
    if not p.exists(): raise BirgerCurveError(f"curve file does not exist: {p}")
    raw=p.read_text(encoding="utf-8-sig")
    if not raw.strip(): raise BirgerCurveError(f"curve CSV is empty: {p}")
    delimiter=_csv_delimiter(raw[:8192])
    decimal_comma=(delimiter != ",")
    rows=[]
    header_seen=False
    data_started=False
    with p.open("r",encoding="utf-8-sig",newline="") as f:
        for lineno,row in enumerate(csv.reader(f,delimiter=delimiter),start=1):
            if not row or not any(str(c).strip() for c in row):
                continue
            first=str(row[0]).strip() if row else ""
            if first.startswith("#"):
                continue
            if len(row)!=2:
                raise BirgerCurveError(
                    f"cannot parse line {lineno} of {p}: expected exactly two columns, got {len(row)}; "
                    f"detected delimiter={delimiter!r}"
                )
            try:
                eps=_csv_number(row[0],decimal_comma=decimal_comma)
                sig=_csv_number(row[1],decimal_comma=decimal_comma)
            except Exception as exc:
                if not data_started and not header_seen:
                    # Exactly one leading header row is allowed.  Comments may
                    # precede it, but a second non-numeric row is never hidden.
                    header_seen=True
                    continue
                raise BirgerCurveError(
                    f"cannot parse numeric data at line {lineno} of {p}; "
                    "check the column delimiter and decimal separator. Silent row skipping is forbidden."
                ) from exc
            data_started=True
            rows.append([eps,sig])
    if not rows:
        raise BirgerCurveError(f"curve CSV contains no numeric data points: {p}")
    return _clean_points(rows)


def table_callable(points,extrapolation):
    a=_clean_points(points); x=a[:,0]; y=a[:,1]
    ext=str(extrapolation).upper()
    if ext not in EXTRAPOLATIONS: raise BirgerCurveError(f"extrapolation must be one of {EXTRAPOLATIONS}")
    def phi(eps):
        e=np.asarray(eps,float)
        if np.any(e<0): raise BirgerCurveError("epsilon_i cannot be negative")
        if ext=="ERROR" and (np.any(e<x[0]) or np.any(e>x[-1])):
            raise BirgerCurveError(f"epsilon_i is outside curve range [{x[0]}, {x[-1]}]")
        out=np.interp(e,x,y)
        if ext=="LINEAR":
            lo=e<x[0]; hi=e>x[-1]
            if np.any(lo): out=np.asarray(out); out[lo]=y[0]+(e[lo]-x[0])*(y[1]-y[0])/(x[1]-x[0])
            if np.any(hi): out=np.asarray(out); out[hi]=y[-1]+(e[hi]-x[-1])*(y[-1]-y[-2])/(x[-1]-x[-2])
        if not np.all(np.isfinite(out)) or np.any(out<0):
            raise BirgerCurveError("curve interpolation/extrapolation returned invalid stress")
        return out
    phi.points=a
    return phi


def piecewise_callable(segments,parameters=None):
    if not isinstance(segments,list) or not segments:
        raise BirgerCurveError("piecewise curve needs at least one segment")
    compiled=[]; prev=-np.inf
    for seg in segments:
        if not isinstance(seg,dict): raise BirgerCurveError("piecewise segment must be an object")
        mx=seg.get("epsilon_max")
        mx=np.inf if mx is None else float(mx)
        if mx<=prev: raise BirgerCurveError("piecewise epsilon_max values must increase")
        compiled.append((mx,formula_callable(seg.get("formula",""),parameters)))
        prev=mx
    if not np.isinf(compiled[-1][0]):
        raise BirgerCurveError("last piecewise segment must have epsilon_max=null (unbounded)")
    def phi(eps):
        e=np.asarray(eps,float); out=np.empty_like(e)
        lo=-np.inf
        assigned=np.zeros(e.shape,dtype=bool)
        for mx,fn in compiled:
            mask=(e>lo)&(e<=mx) if np.isfinite(lo) else (e<=mx)
            if np.any(mask): out[mask]=fn(e[mask])
            assigned|=mask; lo=mx
        if not np.all(assigned): raise BirgerCurveError("piecewise curve does not cover all epsilon_i")
        return out
    return phi


def build_curve(spec,base_dir=None):
    if not isinstance(spec,dict): raise BirgerCurveError("Birger curve spec must be an object")
    mode=str(spec.get("mode","")).upper()
    if mode=="TABLE":
        return table_callable(spec.get("points"),spec.get("extrapolation","ERROR"))
    if mode=="CSV":
        p=Path(str(spec.get("path","")))
        if not p.is_absolute() and base_dir is not None: p=Path(base_dir)/p
        return table_callable(read_csv_points(p),spec.get("extrapolation","ERROR"))
    if mode=="FORMULA":
        return formula_callable(spec.get("formula",""),spec.get("parameters"))
    if mode=="PIECEWISE":
        return piecewise_callable(spec.get("segments"),spec.get("parameters"))
    raise BirgerCurveError("build_curve supports TABLE/CSV/FORMULA/PIECEWISE; EXTERNAL_PYTHON is handled by dispatcher")


def sample_curve(spec,base_dir=None,epsilon_max=None,n=300):
    fn=build_curve(spec,base_dir=base_dir)
    if epsilon_max is None:
        if str(spec.get("mode","")).upper()=="TABLE":
            pts=_clean_points(spec.get("points")); epsilon_max=float(pts[-1,0])
        elif str(spec.get("mode","")).upper()=="CSV":
            p=Path(str(spec.get("path",""))); p=p if p.is_absolute() or base_dir is None else Path(base_dir)/p
            epsilon_max=float(read_csv_points(p)[-1,0])
        else:
            raise BirgerCurveError("epsilon_max is required to preview formula/piecewise curve")
    x=np.linspace(0.0,float(epsilon_max),int(n)); y=fn(x)
    return x,y
