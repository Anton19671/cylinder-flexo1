#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast, calculation-free consistency checks for the capability registry."""
import final_capability_registry as registry


def main():
    rows = registry.capability_table()
    assert len(rows) == 270
    assert {row["status"] for row in rows} == {registry.SUPPORTED}

    for row in rows:
        assert row["status"] == registry.SUPPORTED
        assert row["full_classical_green_lagrange"]

    assert registry.load_capability(
        "CUF", "UNIFORM_RADIAL_FACE", "OUTER"
    ).status == registry.SUPPORTED
    assert registry.load_capability(
        "CUF", "POINT_RADIAL_FORCE", "MIDDLE"
    ).status == registry.SUPPORTED
    assert registry.load_capability(
        "KL", "PATCH_RADIAL_FACE", "MIDDLE"
    ).status == registry.SUPPORTED
    assert registry.load_capability(
        "KL", "LINE_RADIAL_GENERATOR", "INNER"
    ).status == registry.SUPPORTED

    for theory in registry.THEORIES:
        for load_type in registry.LOAD_TYPES[:-1]:
            for surface in registry.LOAD_SURFACES:
                assert registry.load_capability(
                    theory,load_type,surface
                ).status == registry.SUPPORTED

    cuf_case = {
        "theory": "CUF",
        "cuf_model": "TE4",
        "method": "MVI",
        "rank": 8,
        "mode": "NONLINEAR_BIRGER",
        "geometry_model": "GREEN_LAGRANGE",
        "material_model": "BIRGER",
        "load_definition": {
            "type": "PATCH_RADIAL_FACE",
            "face": "OUTER",
        },
    }
    assert registry.case_capability(cuf_case).status == registry.SUPPORTED

    kl_case = {
        "theory": "KL",
        "method": "FDM",
        "mode": "NONLINEAR_ELASTIC",
        "geometry_model": "GREEN_LAGRANGE",
        "material_model": "HOOKE",
        "load_definition": {
            "type": "UNIFORM_RADIAL_FACE",
            "surface": "MIDDLE",
        },
    }
    assert registry.case_capability(kl_case).status == registry.SUPPORTED
    print("CAPABILITY_REGISTRY_SELF_CHECK: PASS")


if __name__ == "__main__":
    main()
