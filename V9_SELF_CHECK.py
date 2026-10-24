#!/usr/bin/env python3
from __future__ import annotations
import json, tempfile
from pathlib import Path
import numpy as np
import final_birger_curve as bc


def _write(root,name,text):
    p=Path(root)/name
    p.write_text(text,encoding='utf-8')
    return p


def run_checks(verbose=True):
    rep={"status":"PASS"}
    with tempfile.TemporaryDirectory(prefix='v9_csv_') as td:
        # Standard comma CSV with one leading header + comments.
        p=_write(td,'comma.csv','# Birger intensity curve\nepsilon_i,Phi\n0.0,0.0\n0.001,242.3\n0.002,300.0\n')
        a=bc.read_csv_points(p)
        assert a.shape==(3,2) and np.allclose(a[-1],[0.002,300.0])
        rep['comma_header_comments']='PASS'

        # Excel/Russian-style semicolon delimiter with decimal comma.
        p=_write(td,'semicolon.csv','epsilon_i;Phi\n0,0;0,0\n0,001;242,3\n0,002;300,0\n')
        a=bc.read_csv_points(p)
        assert np.allclose(a[:,0],[0.0,0.001,0.002]) and np.allclose(a[:,1],[0.0,242.3,300.0])
        rep['semicolon_decimal_comma']='PASS'

        # Tab-separated input is accepted too.
        p=_write(td,'tab.csv','eps\tPhi\n0\t0\n0.001\t242.3\n0.002\t300\n')
        assert bc.read_csv_points(p).shape==(3,2)
        rep['tab']='PASS'

        # A malformed row inside data must name its physical line.
        p=_write(td,'bad_numeric.csv','eps,Phi\n0,0\n0.001,242.3\n0.002,not_a_number\n0.003,350\n')
        try:
            bc.read_csv_points(p)
        except bc.BirgerCurveError as exc:
            msg=str(exc)
            assert 'line 4' in msg and 'Silent row skipping' in msg
            rep['bad_numeric_rejected']=msg
        else:
            raise AssertionError('V9 failed: malformed numeric row was silently accepted')

        # A second header/non-numeric record is forbidden even before data.
        p=_write(td,'two_headers.csv','epsilon,Phi\nstrain,stress\n0,0\n0.001,242.3\n')
        try:
            bc.read_csv_points(p)
        except bc.BirgerCurveError as exc:
            assert 'line 2' in str(exc)
            rep['second_header_rejected']=str(exc)
        else:
            raise AssertionError('V9 failed: second header row was silently accepted')

        # Wrong column count is explicit, not an IndexError/skip.
        p=_write(td,'three_cols.csv','eps;Phi\n0;0\n0.001;242.3;EXTRA\n0.002;300\n')
        try:
            bc.read_csv_points(p)
        except bc.BirgerCurveError as exc:
            assert 'line 3' in str(exc) and 'exactly two columns' in str(exc)
            rep['column_count_rejected']=str(exc)
        else:
            raise AssertionError('V9 failed: wrong column count was accepted')

    if verbose:
        print('V9_SELF_CHECK: PASS')
        print('  comma/semicolon/TAB delimiters: PASS')
        print('  decimal comma with semicolon: PASS')
        print('  malformed numeric row reports line number: PASS')
        print('  only one leading header row allowed: PASS')
        print('  wrong column count: REJECTED')
        print('V9_REPORT_JSON='+json.dumps(rep,ensure_ascii=False,sort_keys=True))
    return rep

if __name__=='__main__':
    run_checks(True)
