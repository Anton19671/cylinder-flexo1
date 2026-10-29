"""Read and write the physical CASE in JSON, YAML or YML.

The solver works with a single mapping in memory. Only transport syntax is
selected by filename extension; no physical fields or defaults are changed.
"""
from __future__ import annotations

import json
from pathlib import Path

CASE_FILETYPES = (
    ("CASE (JSON и YAML)", "*.json *.yaml *.yml"),
    ("JSON", "*.json"),
    ("YAML", "*.yaml *.yml"),
    ("Все файлы", "*.*"),
)
_YAML_SUFFIXES = {".yaml", ".yml"}


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Для загрузки YAML нужен PyYAML. Установите зависимости "
            "из requirements.txt (python -m pip install -r requirements.txt)."
        ) from exc
    return yaml


def load_case_file(file_path):
    """Load an explicit, top-level mapping without changing the case schema."""
    path = Path(file_path).expanduser()
    suffix = path.suffix.lower()
    if suffix not in (".json", ".yaml", ".yml"):
        raise ValueError(f"Формат CASE не поддерживается: {suffix}. Ожидается JSON/YAML/YML.")
    raw = path.read_text(encoding="utf-8-sig")
    if suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ошибка JSON в {path}: {exc}") from exc
    else:
        yaml = _yaml_module()
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Ошибка YAML в {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("CASE должен содержать один объект с полями расчёта (mapping).")
    if not all(isinstance(key, str) for key in data):
        raise ValueError("Ключи верхнего уровня CASE должны быть строками.")
    return data


def save_case_file(file_path, case):
    """Serialize validated CASE in its selected format; do not change physics."""
    path = Path(file_path).expanduser()
    if not isinstance(case, dict):
        raise ValueError("CASE должен быть словарём")
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.dumps(case, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    elif suffix in _YAML_SUFFIXES:
        yaml = _yaml_module()
        payload = yaml.safe_dump(case, allow_unicode=True, sort_keys=False)
    else:
        raise ValueError("Для сохранения CASE выберите расширение .json, .yaml или .yml.")
    path.write_text(payload, encoding="utf-8")
    return path


__all__ = ("CASE_FILETYPES", "load_case_file", "save_case_file")
