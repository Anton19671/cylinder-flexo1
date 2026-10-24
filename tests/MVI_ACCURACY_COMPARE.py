from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def _step_npz_files(root: Path):
    files = sorted(root.glob('STEP_*_STATE.npz'))
    if files:
        return files
    return sorted(p for p in root.glob('STEP_*.npz') if 'NOT_CONVERGED' not in p.name)


def _safe_rel(diff: np.ndarray, ref: np.ndarray) -> float:
    den = float(np.max(np.abs(ref))) if ref.size else 0.0
    num = float(np.max(np.abs(diff))) if diff.size else 0.0
    return num / max(den, np.finfo(float).tiny)


def _compare_npz(a: Path, b: Path):
    rows = []
    with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
        common = sorted(set(za.files) & set(zb.files))
        only_a = sorted(set(za.files) - set(zb.files))
        only_b = sorted(set(zb.files) - set(za.files))
        for key in common:
            xa = np.asarray(za[key])
            xb = np.asarray(zb[key])
            if xa.shape != xb.shape:
                rows.append({
                    'array': key, 'shape_baseline': list(xa.shape),
                    'shape_optimized': list(xb.shape), 'shape_match': False,
                })
                continue
            if not (np.issubdtype(xa.dtype, np.number) and np.issubdtype(xb.dtype, np.number)):
                rows.append({'array': key, 'shape_match': True, 'numeric': False})
                continue
            xa = xa.astype(float, copy=False)
            xb = xb.astype(float, copy=False)
            d = xb - xa
            rows.append({
                'array': key,
                'shape_match': True,
                'numeric': True,
                'max_abs_diff': float(np.max(np.abs(d))) if d.size else 0.0,
                'max_rel_diff_vs_baseline_scale': _safe_rel(d, xa),
                'baseline_abs_max': float(np.max(np.abs(xa))) if xa.size else 0.0,
                'optimized_abs_max': float(np.max(np.abs(xb))) if xb.size else 0.0,
            })
    return rows, only_a, only_b


def _find_json_for_npz(p: Path):
    stem = p.stem.replace('_STATE', '')
    q = p.with_name(stem + '.json')
    return q if q.exists() else None


def _json_controls(p: Path | None):
    if p is None:
        return {}
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}
    keys = (
        'converged', 'solver_status', 'status', 'solver_iterations',
        'iterations', 'xy_cycles', 'projected_inf', 'projected_relative',
        'field_increment_rel', 'load_factor', 'step_index'
    )
    return {k: d.get(k) for k in keys if k in d}


def compare_dirs(base: Path, opt: Path):
    af = _step_npz_files(base)
    bf = _step_npz_files(opt)
    amap = {p.name.replace('_STATE', ''): p for p in af}
    bmap = {p.name.replace('_STATE', ''): p for p in bf}
    common = sorted(set(amap) & set(bmap))
    report = {
        'baseline_dir': str(base.resolve()),
        'optimized_dir': str(opt.resolve()),
        'baseline_steps': len(af),
        'optimized_steps': len(bf),
        'common_steps': len(common),
        'steps': [],
    }
    worst_abs = 0.0
    worst_rel = 0.0
    worst_abs_where = None
    worst_rel_where = None
    for name in common:
        a, b = amap[name], bmap[name]
        rows, only_a, only_b = _compare_npz(a, b)
        for r in rows:
            if not r.get('numeric'):
                continue
            if r['max_abs_diff'] > worst_abs:
                worst_abs = r['max_abs_diff']; worst_abs_where = f'{name}:{r["array"]}'
            if r['max_rel_diff_vs_baseline_scale'] > worst_rel:
                worst_rel = r['max_rel_diff_vs_baseline_scale']; worst_rel_where = f'{name}:{r["array"]}'
        report['steps'].append({
            'step': name,
            'baseline_file': a.name,
            'optimized_file': b.name,
            'arrays': rows,
            'only_baseline_arrays': only_a,
            'only_optimized_arrays': only_b,
            'baseline_controls': _json_controls(_find_json_for_npz(a)),
            'optimized_controls': _json_controls(_find_json_for_npz(b)),
        })
    report['summary'] = {
        'worst_max_abs_diff': worst_abs,
        'worst_max_abs_diff_at': worst_abs_where,
        'worst_scaled_rel_diff': worst_rel,
        'worst_scaled_rel_diff_at': worst_rel_where,
        'same_step_count': len(af) == len(bf) == len(common),
    }
    return report


def main():
    ap = argparse.ArgumentParser(description='Compare baseline and optimized MVI result folders; does not run the solver.')
    ap.add_argument('baseline', type=Path)
    ap.add_argument('optimized', type=Path)
    ap.add_argument('-o', '--output', type=Path, default=Path('MVI_ACCURACY_REPORT.json'))
    ns = ap.parse_args()
    report = compare_dirs(ns.baseline, ns.optimized)
    ns.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    s = report['summary']
    print('MVI accuracy comparison')
    print(f"common steps: {report['common_steps']}")
    print(f"worst max abs diff: {s['worst_max_abs_diff']:.17g} at {s['worst_max_abs_diff_at']}")
    print(f"worst scaled rel diff: {s['worst_scaled_rel_diff']:.17g} at {s['worst_scaled_rel_diff_at']}")
    print(f"same step count: {s['same_step_count']}")
    print(f"report: {ns.output}")


if __name__ == '__main__':
    main()
