"""CUF supports with thickness-dependent constraints cannot be flattened."""
import pytest
from final_boundary_registry import cuf_kinds, cuf_kinds_tau


def test_cuf_simply_supported_requires_tau():
    with pytest.raises(ValueError, match="cuf_kinds_tau"):
        cuf_kinds("SIMPLY_SUPPORTED")


def test_cuf_simply_supported_by_tau():
    assert cuf_kinds_tau("SIMPLY_SUPPORTED", 0) == (
        "GEOMETRIC", "GEOMETRIC", "GEOMETRIC")
    assert cuf_kinds_tau("SIMPLY_SUPPORTED", 1) == (
        "NATURAL", "NATURAL", "NATURAL")
    assert cuf_kinds_tau("SIMPLY_SUPPORTED", 2) == (
        "NATURAL", "NATURAL", "NATURAL")
