#!/usr/bin/env python3
"""Consolidate actual CUF TE1..TE4 MVI-1..8 GitHub Actions results.

Only real, complete, mode-and-mass-MAC checked CI results can become PASS.
The current physical geometry/material is Rawat (2016), not Liang et al.
(2023) Carrera Table 10. Both literature sources must remain distinct.
The source operator is shared by MVI/FDM, not a separate 3D validation.
"""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path

MODELS = ("CUF_TE1", "CUF_TE2", "CUF_TE3", "CUF_TE4")
RANKS = tuple(range(1, 9))
N_MODES = 8
SOURCE_NOTE = (
    "CUF Taylor TE1-TE4 on Rawat 2016 isotropic C-C shell: "
    "L=2m R=1m h=0.05m E=210GPa nu=0.3 rho=7850kg/m3. "
    "Same SBP42 13x13 discrete physics for MVI/FDM. "
    "NOT an independent Carrera Table 10 verification."
)


def _safe_numbers(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError("missing or incorrectly sized frequency array")
    numbers = [float(x) for x in value]
    if any(not math.isfinite(x) or x <= 0 for x in numbers):
        raise ValueError("invalid or nonpositive frequency array")
    if any(numbers[i] > numbers[i + 1] + 1e-7
           for i in range(size - 1)):
        raise ValueError("unsorted frequency array")
    return numbers


def summarize(root):
    root = Path(root)
    rows = []
    payload = {"source": SOURCE_NOTE,
               "independent_carrera_validation": False,
               "experimental_status": "UNVERIFIED",
               "cases": {}, "rows": [], "passed": 0, "total": 32}
    for model in MODELS:
        groups = {}
        model_fdm = None
        for group_name, ranks in (("low", (1, 2, 3, 4)),
                                  ("high", (5, 6, 7, 8))):
            folder = root / f"carrera-{model}-{group_name}"
            filename = folder / "group_summary.json"
            if filename.exists():
                try:
                    group = json.loads(filename.read_text(encoding="utf-8"))
                    if (group.get("theory") != model or
                            group.get("ranks") != list(ranks) or
                            group.get("grid") != [13, 13] or
                            group.get("boundary") != "C-C" or
                            group.get("n_modes") != N_MODES or
                            group.get("fdm_same_discretization_not_independent_physics")
                                is not True):
                        raise ValueError("group identity, physical scope or method mismatch")
                except (ValueError, OSError, TypeError) as exc:
                    group = {"invalid_group": str(exc), "runs": {}}
            else:
                group = {"missing_group": str(filename), "runs": {}}
            groups[group_name] = group
            fdm_rec = group.get("runs", {}).get("FDM", {})
            if fdm_rec.get("status") == "SOLVED":
                try:
                    this_fdm = _safe_numbers(fdm_rec.get("frequencies_hz"),
                                             N_MODES)
                    if model_fdm is None:
                        model_fdm = this_fdm
                    elif max(abs(x - y) for x, y in
                             zip(model_fdm, this_fdm)) > 1e-5:
                        group["invalid_group"] = "LOW_HIGH_FDM_REFERENCE_MISMATCH"
                except (ValueError, TypeError) as exc:
                    group["invalid_group"] = str(exc)

        payload["cases"][model] = {"FDM_hz": model_fdm,
                                   "groups": groups}
        for rank in RANKS:
            group_name = "low" if rank <= 4 else "high"
            group = groups[group_name]
            record = group.get("runs", {}).get(f"MVI-{rank}", {})
            status = "MISSING"
            error = ""
            freqs = None
            fdm = None
            cmp = record.get("comparison_with_fdm", {})
            if group.get("invalid_group"):
                status, error = "FAIL", group["invalid_group"]
            elif group.get("missing_group"):
                error = group["missing_group"]
            elif record.get("status") != "SOLVED":
                status = "ERROR" if record else "MISSING"
                error = record.get("error", "No MVI record")
            else:
                try:
                    freqs = _safe_numbers(record.get("frequencies_hz"), N_MODES)
                    fdm = _safe_numbers(group.get("runs", {})
                                        .get("FDM", {}).get("frequencies_hz"),
                                        N_MODES)
                    if cmp.get("status") == "PASS":
                        if (cmp.get("frequency_verified") is not True or
                                cmp.get("all_mode_subspaces_verified") is not True or
                                cmp.get("issues") != []):
                            raise ValueError("false PASS comparison flags")
                        groups_mac = cmp.get("mode_subspaces", [])
                        visited = [i for g in groups_mac for i in g.get("modes", [])]
                        minima = [float(g.get("minimum_mass_mac", 0))
                                  for g in groups_mac]
                        if (sorted(visited) != list(range(1, 9)) or
                                not minima or min(minima) < 0.999):
                            raise ValueError("missing or mismatched physical modal subspaces")
                        status = "PASS"
                    else:
                        status = "FAIL"
                        error = (cmp.get("error") or
                                 "; ".join(cmp.get("issues", [])) or
                                 "frequency or modal-subspace mismatch")
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    status, error = "FAIL", str(exc)

            if status == "PASS":
                payload["passed"] += 1
            row = {
                "model": model, "rank": rank,
                "grid": "13x13", "bc": "C-C",
                "status": status, "error": error,
                "fdm_first_frequency_hz": "" if not fdm else f"{fdm[0]:.14g}",
                "mvi_first_frequency_hz": "" if not freqs else f"{freqs[0]:.14g}",
                "max_frequency_error_relative": cmp.get(
                    "maximum_relative_frequency_error", ""),
                "minimum_mass_mac": (
                    min((float(g.get("minimum_mass_mac", 0))
                         for g in cmp.get("mode_subspaces", [])), default="")
                    if isinstance(cmp, dict) else ""),
                "mvi_sweeps_per_mode": json.dumps(record.get("mvi_sweeps", [])),
                "mvi_wall_seconds": record.get("wall_seconds", ""),
                "fdm_wall_seconds": group.get("runs", {}).get(
                    "FDM", {}).get("wall_seconds", ""),
                "fdm_eight_hz": json.dumps(fdm) if fdm else "",
                "mvi_eight_hz": json.dumps(freqs) if freqs else "",
                "spectrum_complete": record.get(
                    "global_spectrum_completeness_verified", False),
            }
            rows.append(row)
    payload["rows"] = rows
    payload["status"] = ("PASS" if payload["passed"] == payload["total"]
                         else "INCOMPLETE_OR_FAILED")
    return payload


def _word(payload, output):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt

    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(29.7), Cm(21.0)
    section.top_margin = section.bottom_margin = Cm(1.5)
    section.left_margin = section.right_margin = Cm(1.5)
    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(12)
    for level, size in (("Title", 16), ("Heading 1", 14), ("Heading 2", 12)):
        styles[level].font.name = "Times New Roman"
        styles[level].font.size = Pt(size)
        styles[level].font.color.rgb = None
    doc.add_heading("Модальный CUF TE1–TE4: МВИ-1…8 и МКР/FDM", 0)
    doc.add_paragraph("Сводная проверка на одной дискретной задаче, 13×13, C–C.")
    doc.add_paragraph(SOURCE_NOTE)
    doc.add_paragraph(
        f"Подтверждено {payload['passed']} из {payload['total']} сочетаний. "
        f"Итоговый статус: {payload['status']}. Статус PASS означает совпадение "
        "частот и массовых модальных подпространств; он не подтверждает "
        "физическую точность по независимому 3D-эталону."
    )
    doc.add_heading("Состояние всех сочетаний", 1)
    table = doc.add_table(rows=1, cols=9)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for c, s in zip(table.rows[0].cells, ["Теория"] +
                    [f"МВИ-{i}" for i in range(1, 9)]):
        c.text = s
    for model in MODELS:
        row = table.add_row().cells
        row[0].text = model.replace("CUF_", "")
        for i, record in enumerate((r for r in payload["rows"]
                                    if r["model"] == model), start=1):
            row[i].text = record["status"]
    for row in table.rows:
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading("Расчётные частоты по восьми формам", 1)
    for model in MODELS:
        doc.add_heading(model.replace("CUF_", "CUF "), 2)
        fdm = payload["cases"][model]["FDM_hz"]
        doc.add_paragraph(
            "Частоты в Гц. Знак «—» означает отсутствие подтверждённого результата."
        )
        rmodel = [r for r in payload["rows"] if r["model"] == model]
        tb = doc.add_table(rows=1, cols=10)
        tb.style = "Table Grid"
        headers = ["№", "FDM"] + [f"МВИ-{k}" for k in RANKS]
        for cell, label in zip(tb.rows[0].cells, headers):
            cell.text = label
        allfreq = {}
        for r in rmodel:
            if r["mvi_eight_hz"] and r["status"] == "PASS":
                allfreq[r["rank"]] = json.loads(r["mvi_eight_hz"])
        for k in range(8):
            row = tb.add_row().cells
            row[0].text = str(k + 1)
            row[1].text = "—" if not fdm else f"{fdm[k]:.5f}"
            for rank in RANKS:
                seq = allfreq.get(rank)
                row[rank + 1].text = "—" if seq is None else f"{seq[k]:.5f}"
        doc.add_paragraph(
            "Сопоставление с опубликованной работой Liang et al. (2023), "
            "Materials 16, 3345, Table 10, требует отдельной постановки "
            "с h=0,1 м и E=206 ГПа; эти частоты сюда не подставляются."
        )
    doc.add_heading("Ограничения", 1)
    doc.add_paragraph(
        "Приведённые данные проверяют алгоритм МВИ на той же дискретной "
        "физической модели, что и МКР/FDM. Результаты Rawat et al. (2016) "
        "и Liang et al. (2023) не смешиваются. Для литературной "
        "валидации нужно совпадение геометрии, материальных параметров, "
        "граничных условий, нормировки и модальных индексов."
    )
    doc.save(output)


def main():
    root = Path(os.environ.get("CARRERA_ARTIFACT_DIR",
                               "benchmark_carrera_downloads"))
    out = Path("benchmark_carrera_consolidated")
    out.mkdir(parents=True, exist_ok=True)
    data = summarize(root)
    (out / "matrix32.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    rows = data["rows"]
    with (out / "matrix32.csv").open("w", newline="",
                                    encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# CUF TE1-TE4 / MVI-1..8: verified modal spectrum",
        "", f"Status: **{data['status']}**, {data['passed']}/32 PASS.",
        "", "Scope: " + SOURCE_NOTE, "",
        "| Theory | " + " | ".join(f"MVI-{i}" for i in RANKS) + " |",
        "| --- | " + " | ".join("---" for _ in RANKS) + " |",
    ]
    for model in MODELS:
        lines.append("| " + model + " | " + " | ".join(
            row["status"] for row in rows if row["model"] == model) + " |")
    lines.extend(["", "Rows also available in matrix32.csv with all eight",
                  "frequencies, frequencies differences, mass-MAC and sweep counts.",
                  "This is NOT an independent Carrera 2023 3D validation.", ""])
    (out / "matrix32.md").write_text("\n".join(lines), encoding="utf-8")
    _word(data, out / "CUF_TE1_TE4_MVI1_8_VERIFIED_RU.docx")
    print("[CARRERA-CUF-32] " + json.dumps({
        "status": data["status"], "passed": data["passed"],
        "total": data["total"], "output": str(out)
    }), flush=True)
    if data["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
