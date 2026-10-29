#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single source of truth for CYL_SOLVER supported combinations.

The registry is deliberately independent of the GUI and mechanics modules.
Both the dispatcher and the interface must query this module instead of
maintaining separate, potentially contradictory lists of capabilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Dict, Iterable, Optional, Tuple


SUPPORTED = "SUPPORTED"
LIMITED = "LIMITED"
UNAVAILABLE = "UNAVAILABLE"

THEORIES = ("CUF", "KL")
CUF_MODELS = ("TE1", "TE2", "TE3", "TE4")
GEOMETRY_MODELS = ("LINEAR", "VON_KARMAN", "GREEN_LAGRANGE")
# The flexoelectric model is elastic; Birger/J2 are retired from active routes.
MATERIAL_MODELS = ("HOOKE",)
SCALE_MODELS = ("MACRO", "MICRO")
# Registered for staged implementation, NOT available for numerical runs.
PLANNED_THEORIES = ("FSDT", "REDDY")
THEORY_LABELS = {
    "CUF": "Carrera CUF (TE1-TE4)",
    "KL": "Kirchhoff-Love",
    "FSDT": "Timoshenko/Mindlin FSDT",
    "REDDY": "Reddy TSDT",
}
METHODS = ("FDM", "MVI")
MVI_RANKS = tuple(range(1, 9))
LOAD_TYPES = (
    "UNIFORM_RADIAL_FACE",
    "HARMONIC_RADIAL_FACE",
    "PATCH_RADIAL_FACE",
    "LINE_RADIAL_GENERATOR",
    "POINT_RADIAL_FORCE",
    "EXTERNAL_NPZ",
)
LOAD_SURFACES = ("MIDDLE", "INNER", "OUTER")


@dataclass(frozen=True)
class Capability:
    status: str
    code: str
    message_ru: str
    full_classical_green_lagrange: bool

    @property
    def can_run(self) -> bool:
        return self.status in (SUPPORTED, LIMITED)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _upper(value: Any) -> str:
    return str(value or "").strip().upper()


def _combine(items: Iterable[Capability]) -> Capability:
    values = tuple(items)
    priority = {SUPPORTED: 0, LIMITED: 1, UNAVAILABLE: 2}
    worst = max(values, key=lambda item: priority[item.status])
    messages = []
    codes = []
    for item in values:
        if item.status != SUPPORTED and item.message_ru not in messages:
            messages.append(item.message_ru)
            codes.append(item.code)
    if not messages:
        return worst
    return Capability(
        status=worst.status,
        code="+".join(codes),
        message_ru=" ".join(messages),
        full_classical_green_lagrange=all(
            item.full_classical_green_lagrange for item in values
        ),
    )


def physics_capability(
    theory: str,
    geometry_model: str,
    material_model: str,
    method: str,
    *,
    cuf_model: Optional[str] = None,
    rank: Optional[int] = None,
) -> Capability:
    """Return the audited status of one mechanics/method combination."""
    theory = _upper(theory)
    geometry_model = _upper(geometry_model)
    material_model = _upper(material_model)
    method = _upper(method)
    cuf_model = _upper(cuf_model)

    if theory in PLANNED_THEORIES:
        return Capability(
            UNAVAILABLE, "THEORY_PENDING_VARIATIONAL_IMPLEMENTATION",
            f"{THEORY_LABELS[theory]}: кинематическая ветка запланирована; "
            "функционал, граничные условия, МВИ и МКР ещё не верифицированы.",
            False,
        )
    if theory not in THEORIES:
        return Capability(UNAVAILABLE, "THEORY_UNKNOWN", "Неизвестная теория.", False)
    if geometry_model not in GEOMETRY_MODELS:
        return Capability(UNAVAILABLE, "GEOMETRY_UNKNOWN", "Неизвестная геометрическая модель.", False)
    if material_model in ("BIRGER", "J2_INCREMENTAL"):
        return Capability(
            UNAVAILABLE, "MATERIAL_RETIRED_FLEXO",
            "Биргер и J2-пластичность исключены из cylinder-flexo1; только упругий материал.",
            False,
        )
    if material_model not in MATERIAL_MODELS:
        return Capability(UNAVAILABLE, "MATERIAL_UNKNOWN", "Неизвестная модель материала.", False)
    if method not in METHODS:
        return Capability(UNAVAILABLE, "METHOD_UNKNOWN", "Неизвестный численный метод.", False)
    if theory == "CUF" and cuf_model not in CUF_MODELS:
        return Capability(UNAVAILABLE, "CUF_MODEL_UNKNOWN", "Для CUF необходимо выбрать TE1–TE4.", False)
    if method == "MVI" and rank not in MVI_RANKS:
        return Capability(UNAVAILABLE, "MVI_RANK_INVALID", "Ранг МВИ должен быть от 1 до 8.", False)
    return Capability(
        SUPPORTED,
        "FULLY_SUPPORTED",
        "Сочетание полностью поддерживается текущей реализацией.",
        True,
    )


def load_capability(theory: str, load_type: str, surface: Optional[str]) -> Capability:
    """Return status of the load projection for the selected shell theory."""
    theory = _upper(theory)
    load_type = _upper(load_type)
    surface = _upper(surface)
    if load_type not in LOAD_TYPES:
        return Capability(UNAVAILABLE, "LOAD_UNKNOWN", "Неизвестный тип нагрузки.", False)
    if load_type == "EXTERNAL_NPZ":
        return Capability(SUPPORTED, "EXTERNAL_LOAD", "Внешний массив нагрузки проверяется по форме при чтении.", True)
    if surface not in LOAD_SURFACES:
        return Capability(UNAVAILABLE, "LOAD_SURFACE_MISSING", "Необходимо выбрать срединную, внутреннюю или внешнюю поверхность нагрузки.", False)
    if theory not in THEORIES and theory not in PLANNED_THEORIES:
        return Capability(UNAVAILABLE, "THEORY_UNKNOWN", "Неизвестная теория.", False)
    return Capability(
        SUPPORTED,
        "LOAD_SUPPORTED_ALL_SURFACES",
        "Нагрузка поддерживается на срединной, внутренней и внешней поверхностях с точным учётом метрики поверхности.",
        True,
    )


def case_capability(case: Dict[str, Any]) -> Capability:
    """Evaluate physics, method, load, surface and macro/micro route."""
    from cylinder_reference_case import selected_backend, BACKEND, prepare_reference_case
    try:
        backend=selected_backend(case)
        if backend==BACKEND:
            if str(case.get("calculation_kind","")).upper()=="MODAL":
                from cylinder_modal_reference import prepare_modal_case
                prepare_modal_case(case)
                return Capability(LIMITED,"REFERENCE_MODAL",
                    "Проверочные собственные частоты: семь теорий, малая сетка.",False)
            reference=prepare_reference_case(case)
            return Capability(LIMITED,"REFERENCE_"+reference.calculation_kind+"_"+reference.method,
                              "Проверочный вариационный расчёт на малой сетке; сходимость по сетке не подтверждена.",False)
    except (ValueError,TypeError,KeyError) as exc:
        return Capability(UNAVAILABLE,"REFERENCE_CASE_INVALID",str(exc),False)
    scale = _upper(case.get("scale_model", "MACRO"))
    if scale not in SCALE_MODELS:
        return Capability(UNAVAILABLE, "SCALE_MODEL_UNKNOWN",
                          "Допустимы только две физические ветки: MACRO или MICRO.", False)
    if scale == "MICRO":
        micro = case.get("micro")
        flag = micro.get("enable_flexoelectric", True) if isinstance(micro, dict) else True
        if not isinstance(flag, bool):
            return Capability(UNAVAILABLE, "MICRO_FLEXO_SWITCH_INVALID",
                              "Переключатель флексоэффекта должен быть true/false.", False)
        if flag:
            return Capability(UNAVAILABLE, "MICRO_G3D_COUPLED_SOLVER_PENDING",
                              "Связанная механико-электрическая система G3D с высшими ГУ, "
                              "МВИ/МКР и динамикой пока не завершена. Подмена MACRO запрещена.", False)
        return Capability(UNAVAILABLE, "MICRO_G3D_GRADIENT_SOLVER_PENDING",
                          "Флексоэффект отключён (mu1=mu2=kappa=0). Градиентные "
                          "механические уравнения и высшие ГУ ещё не собраны. MACRO-подмена запрещена.", False)
    theory = _upper(case.get("theory"))
    if theory == "CUF_TE1":
        theory = "CUF"
    mode = _upper(case.get("mode"))
    geometry = _upper(case.get("geometry_model"))
    if not geometry:
        geometry = "LINEAR" if mode.startswith("LINEAR") else (
            "GREEN_LAGRANGE" if theory == "CUF" else "VON_KARMAN"
        )
    material = _upper(case.get("material_model")) or (
        "BIRGER" if mode.endswith("BIRGER") else (
            "J2_INCREMENTAL" if mode.endswith("J2") else "HOOKE"
        )
    )
    method = _upper(case.get("method"))
    raw_rank = case.get("rank")
    try:
        rank = int(raw_rank) if method == "MVI" else None
    except (TypeError, ValueError):
        rank = None
    physics = physics_capability(
        theory,
        geometry,
        material,
        method,
        cuf_model=case.get("cuf_model", "TE1" if theory == "CUF" else None),
        rank=rank,
    )
    load = case.get("load_definition") or {}
    load_type = _upper(load.get("type"))
    if not load_type:
        return physics
    surface = load.get("surface", load.get("face"))
    return _combine((physics, load_capability(theory, load_type, surface)))


def capability_table() -> Tuple[Dict[str, Any], ...]:
    """Machine-readable table used by the GUI, diagnostics and audit tests."""
    rows = []
    for theory, geometry, material, method in product(
        THEORIES, GEOMETRY_MODELS, MATERIAL_MODELS, METHODS
    ):
        models = CUF_MODELS if theory == "CUF" else (None,)
        ranks = MVI_RANKS if method == "MVI" else (None,)
        for model, rank in product(models, ranks):
            cap = physics_capability(
                theory,
                geometry,
                material,
                method,
                cuf_model=model,
                rank=rank,
            )
            rows.append({
                "theory": theory,
                "cuf_model": model,
                "geometry_model": geometry,
                "material_model": material,
                "method": method,
                "rank": rank,
                **cap.to_dict(),
            })
    return tuple(rows)


__all__ = (
    "SUPPORTED", "LIMITED", "UNAVAILABLE", "THEORIES", "CUF_MODELS",
    "GEOMETRY_MODELS", "MATERIAL_MODELS", "SCALE_MODELS", "PLANNED_THEORIES", "THEORY_LABELS", "METHODS", "MVI_RANKS",
    "LOAD_TYPES", "LOAD_SURFACES", "Capability", "physics_capability",
    "load_capability", "case_capability", "capability_table",
)

