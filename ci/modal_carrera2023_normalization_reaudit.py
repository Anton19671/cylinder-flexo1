#!/usr/bin/env python3
"""Reassess Liang et al. 2023 Table 10 CUF modal frequencies without solving.

The published normalization is printed as Omega = omega*L*sqrt(rho/E/(2+2nu)),
whose grouping is ambiguous in the typeset caption. Record BOTH:
 A: rho/[E*(2+2nu)] (sequential division);
 B: rho/[E/(2+2nu)] (shear-modulus convention).
Do NOT silently pick the better-fitting convention.

Reads the original actual GitHub Actions numerical outputs. A missing
record remains missing; a green CI job is not an independent 3D validation.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

DATA = Path("config/verification/carrera2023/Liang_Liu_Li_Wang_2023_Table10_CC_R1.csv")
ARTIFACTS = Path(os.environ.get("LIANG_ARTIFACTS", "liang_artifacts"))
OUT = Path("liang2023_normalization_comparison")
THEORIES = ("CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4")
METHODS = ("FDM",) + tuple(f"MVI_{r}" for r in range(1, 9))
E = 206.0e9
RHO = 7850.0
NU = 0.3
L = 2.0

# Omega = 2*pi*f*L*factor; choose the factor ONLY once the literature
# definition is resolved independently, never from agreement with solver.
FACTOR_A = math.sqrt(RHO / (E * (2 + 2 * NU)))
FACTOR_B = math.sqrt(RHO / (E / (2 + 2 * NU)))


def reference_omegas():
    with DATA.open(encoding="utf-8", newline="") as inp:
        lines = list(csv.DictReader(inp))
    if [int(r["mode"]) for r in lines] != list(range(1, 9)):
        raise ValueError("LITERATURE_REFERENCE_MUST_CONTAIN_EIGHT_ORDERED_MODES")
    return [float(r["omega_dimensionless"]) for r in lines]


def main():
    reference = reference_omegas()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for theory in THEORIES:
        for method in METHODS:
            path = ARTIFACTS / f"liang2023-{theory}-{method}" / "result.json"
            record = {}
            if path.exists():
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    record = {"status": "BAD_JSON", "error": str(exc)}
            status = record.get("status", "MISSING")
            frequencies = record.get("frequencies_hz", [])
            if (status == "COMPUTED_COMPARE_ONLY" and isinstance(frequencies, list)
                and len(frequencies) == 8
                and all(isinstance(f, (int, float)) and math.isfinite(f) and f > 0
                        for f in frequencies)):
                for m, (observed, omega_ref) in enumerate(zip(frequencies, reference), 1):
                    norm_A = 2 * math.pi * observed * L * FACTOR_A
                    norm_B = 2 * math.pi * observed * L * FACTOR_B
                    ref_hz_A = omega_ref / (2 * math.pi * L * FACTOR_A)
                    ref_hz_B = omega_ref / (2 * math.pi * L * FACTOR_B)
                    rows.append({
                        "theory": theory, "method": method, "mode": m, "status": status,
                        "frequency_hz_calculated": observed,
                        "omega_published": omega_ref,
                        "A_omega_sequential_division": norm_A,
                        "A_reference_hz": ref_hz_A,
                        "A_difference_pct": (observed / ref_hz_A - 1) * 100,
                        "B_omega_shear_modulus": norm_B,
                        "B_reference_hz": ref_hz_B,
                        "B_difference_pct": (observed / ref_hz_B - 1) * 100,
                        "circumferential_n": (
                            record.get("circumferential_harmonic_n", [None]*8)[m - 1]
                            if len(record.get("circumferential_harmonic_n") or []) == 8
                            else None),
                        "axial_m_verified": False,
                        "literature_mode_mac_available": False,
                        "mvi_sweeps": (
                            (record.get("mvi_sweeps") or [None]*8)[m - 1]
                            if len(record.get("mvi_sweeps") or []) == 8
                            else None),
                        "wall_seconds": record.get("wall_seconds"),
                        "note": "PROVISIONAL_NORMALIZATION_AND_MODE_MATCH",
                    })
            else:
                for m, omega_ref in enumerate(reference, 1):
                    rows.append({
                        "theory": theory, "method": method, "mode": m,
                        "status": status, "frequency_hz_calculated": "",
                        "omega_published": omega_ref,
                        "A_omega_sequential_division": "",
                        "A_reference_hz": omega_ref / (2 * math.pi * L * FACTOR_A),
                        "A_difference_pct": "",
                        "B_omega_shear_modulus": "",
                        "B_reference_hz": omega_ref / (2 * math.pi * L * FACTOR_B),
                        "B_difference_pct": "",
                        "circumferential_n": "",
                        "axial_m_verified": False,
                        "literature_mode_mac_available": False,
                        "mvi_sweeps": "",
                        "wall_seconds": record.get("wall_seconds"),
                        "note": record.get("error", "NO_VERIFIED_FREQUENCIES"),
                    })
    csv_file = OUT / "table10_modes_both_normalizations.csv"
    with csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    done = sorted(set((r["theory"], r["method"]) for r in rows
                      if r["status"] == "COMPUTED_COMPARE_ONLY"))
    summary = {
        "source": "Liang, Liu, Li, Wang, Materials 2023, 16, 3345, Table 10 CC R=1",
        "doi": "10.3390/ma16093345",
        "equation_as_printed": "Omega=omega*L*sqrt(rho/E/(2+2nu))",
        "normalization_interpretation": "UNRESOLVED",
        "A_interpretation": "sqrt(rho/[E*(2+2nu)])",
        "B_interpretation": "sqrt(rho/[E/(2+2nu)])",
        "A_hz_per_published_omega": 1 / (2 * math.pi * L * FACTOR_A),
        "B_hz_per_published_omega": 1 / (2 * math.pi * L * FACTOR_B),
        "validation_level": "PROVISIONAL_COMPARISON_NOT_INDEPENDENTLY_CONFIRMED",
        "mode_axial_identity": "NOT_VERIFIED",
        "total_solver_cases": len(THEORIES)*len(METHODS),
        "completed_solver_cases": len(done),
        "completed_list": done,
        "rows": rows,
    }
    (OUT / "comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    with (OUT / "comparison.md").open("w", encoding="utf-8") as f:
        f.write("# Carrera 2023 CUF: normalization and partial CI comparison\n\n")
        f.write("Table 10 is printed with successive divisions in the normalization. "
                "The two interpretations differ by a factor (2+2nu)=2.6. "
                "Neither should be chosen by fitting to the solver.\n\n")
        f.write(f"Data available for {len(done)}/{len(THEORIES)*len(METHODS)} "
                "solved theory/method cases.\n\n")
        f.write("| Theory | Method | f1 (Hz) | error A (%) | error B (%) |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for r in rows:
            if r["mode"] != 1:
                continue
            def show(v):
                return "—" if v == "" else f"{v:.5f}"
            f.write(f"| {r['theory']} | {r['method']} | "
                    f"{show(r['frequency_hz_calculated'])} | "
                    f"{show(r['A_difference_pct'])} | "
                    f"{show(r['B_difference_pct'])} |\n")
        f.write("\nThe percentages are provisional: the article's normalization, "
                "mode identity and the cross-theory physical scope must be verified. "
                "These values do not constitute an independent validation PASS.\n")
    print("[LIANG-NORMALIZATION] " + json.dumps({
        "available": len(done), "total": len(THEORIES)*len(METHODS),
        "A_hz_per_Omega": summary["A_hz_per_published_omega"],
        "B_hz_per_Omega": summary["B_hz_per_published_omega"],
        "status": summary["validation_level"]}), flush=True)


if __name__ == "__main__":
    main()
