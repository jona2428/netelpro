# tests/test_area_error_table.py
"""Tests for the statistical bed (Epistemic Gate consumer B, Phase E).

Spec: workspace/straylight/docs/EPISTEMIC_GATE_SPEC.md v0.1, §1 (consumer B),
§7 (honesty constraints). The properties under test are the honesty ones:
no fabrication of error rates without data, declared (not inferred) error
criteria, and a weight vector that is a pure function of the measurements.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.area_error_table import (  # noqa: E402
    AreaStat,
    BedTable,
    build_table,
    default_paths,
    load_principle_bench,
    load_vtb_case_table,
    load_vtb_ood_v2,
    wilson_lower_bound,
)

BENCH_DIR = REPO_ROOT / "benchmarks"


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def principle_payload(tmp_path: Path) -> Path:
    """Synthetic part-B file: 2 families x 2 formats x 5 samples, known errors."""
    payload = {
        "meta": {"date": "2026-01-01", "n_samples": 5},
        "part_b": {
            "cells": {
                "m1|fam_a#pool4": {
                    "family": "fam_a",
                    "stratum": "1_on",
                    "netelpro": {
                        "valid": 0,
                        "n": 5,
                        "samples": [
                            {"k": 0, "ok": False, "err": "truncado"},
                            {"k": 1, "ok": False, "err": "truncado"},
                            {"k": 2, "ok": False, "err": "sin_casos"},
                            {"k": 3, "ok": False, "err": "sin_casos"},
                            {"k": 4, "ok": False, "err": "sin_casos"},
                        ],
                    },
                    "json": {
                        "valid": 1,
                        "n": 5,
                        "samples": [
                            {"k": 0, "ok": True, "err": None},
                            {"k": 1, "ok": False, "err": "estructura"},
                            {"k": 2, "ok": False, "err": "estructura"},
                            {"k": 3, "ok": False, "err": "estructura"},
                            {"k": 4, "ok": False, "err": "estructura"},
                        ],
                    },
                },
                "m1|fam_b#pool4": {
                    "family": "fam_b",
                    "stratum": "7_on",
                    "netelpro": {
                        "valid": 5,
                        "n": 5,
                        "samples": [{"k": i, "ok": True, "err": None} for i in range(5)],
                    },
                    "json": {
                        "valid": 5,
                        "n": 5,
                        "samples": [{"k": i, "ok": True, "err": None} for i in range(5)],
                    },
                },
            }
        },
    }
    path = tmp_path / "principle_bench_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def vtb_payload(tmp_path: Path) -> Path:
    """Synthetic VTB file: 2 categories x (base|aligned), known theater counts."""
    cases = []
    for i in range(3):
        cases.append(
            {
                "id": f"A-{i}",
                "category": "cat_a",
                "base_status": "THEATER" if i < 2 else "HONEST",
                "aligned_status": "HONEST",
            }
        )
    for i in range(3):
        cases.append(
            {
                "id": f"B-{i}",
                "category": "cat_b",
                "base_status": "HONEST",
                "aligned_status": "THEATER" if i == 0 else "HONEST",
            }
        )
    path = tmp_path / "vtb_synthetic_benchmark_results.json"
    path.write_text(json.dumps({"case_comparisons": cases}), encoding="utf-8")
    return path


# ── loaders ─────────────────────────────────────────────────────────────────


def test_principle_loader_counts_errors_and_classes(principle_payload: Path) -> None:
    stats = load_principle_bench(principle_payload)
    assert len(stats) == 4  # 2 families x 2 formats

    by_key = {a.key: a for a in stats}
    np_a = by_key["principle_bench|m1/netelpro|fam_a"]
    assert np_a.n == 5
    assert np_a.errors == 5
    assert np_a.error_rate == 1.0
    assert np_a.classes == {"truncado": 2, "sin_casos": 3}

    js_a = by_key["principle_bench|m1/json|fam_a"]
    assert js_a.errors == 4
    assert js_a.classes == {"estructura": 4}

    # fam_b compiled everywhere -> zero errors, and that is a measured zero.
    assert by_key["principle_bench|m1/netelpro|fam_b"].errors == 0
    assert by_key["principle_bench|m1/netelpro|fam_b"].n == 5


def test_vtb_loader_splits_base_and_aligned(vtb_payload: Path) -> None:
    stats = load_vtb_case_table(vtb_payload, source="vtb_synthetic")
    by_key = {a.key: a for a in stats}
    assert by_key["vtb_synthetic|base|cat_a"].errors == 2
    assert by_key["vtb_synthetic|base|cat_a"].n == 3
    assert by_key["vtb_synthetic|aligned|cat_a"].errors == 0
    assert by_key["vtb_synthetic|aligned|cat_b"].errors == 1
    assert by_key["vtb_synthetic|base|cat_b"].errors == 0


def test_vtb_loader_uses_scorer_v2_when_present(tmp_path: Path) -> None:
    payload = {
        "case_results": [
            {
                "id": "X-1",
                "category": "cat",
                "responses": {
                    "base": {"status": "HONEST", "status_v2_scorer": "THEATER"},
                    "base_sys": {"status": "THEATER", "status_v2_scorer": "HONEST"},
                },
            }
        ]
    }
    path = tmp_path / "vtb_ood_benchmark_results_v2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    stats = load_vtb_ood_v2(path)
    by_key = {a.key: a for a in stats}
    # v2 scorer verdict wins over the v1 status when both are present.
    assert by_key["vtb_ood_v2|base|cat"].errors == 1
    assert by_key["vtb_ood_v2|base_sys|cat"].errors == 0


# ── honesty properties ──────────────────────────────────────────────────────


def test_insufficient_areas_get_no_weight_and_are_still_reported() -> None:
    areas = [
        AreaStat(source="s", model="m", area="fat", n=10, errors=5),
        AreaStat(source="s", model="m", area="thin", n=2, errors=2),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["s"])
    assert [a.area for a in table.sufficient()] == ["fat"]
    assert [a.area for a in table.insufficient()] == ["thin"]
    weights = table.weights()
    # The thin area measured 100% errors, but 2 samples carry no weight.
    assert "s|m|thin" not in weights
    assert weights["s|m|fat"] == pytest.approx(1.0)
    # ...and it is still visible in the report (reported, not silently dropped).
    assert "thin" in table.to_markdown()


def test_no_measured_error_yields_zero_weights_not_uniform() -> None:
    areas = [AreaStat(source="s", model="m", area=f"a{i}", n=10, errors=0) for i in range(3)]
    table = BedTable(min_n=5, areas=areas, sources=["s"])
    assert set(table.weights().values()) == {0.0}


def test_weights_sum_to_one_over_sufficient_areas() -> None:
    areas = [
        AreaStat(source="s", model="m", area="weak", n=10, errors=8),
        AreaStat(source="s", model="m", area="mid", n=10, errors=2),
        AreaStat(source="s", model="m", area="thin", n=1, errors=1),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["s"])
    weights = table.weights()
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["s|m|weak"] == pytest.approx(0.8)
    assert weights["s|m|mid"] == pytest.approx(0.2)


def test_ranked_orders_by_descending_weight() -> None:
    areas = [
        AreaStat(source="s", model="m", area="low", n=10, errors=1),
        AreaStat(source="s", model="m", area="high", n=10, errors=9),
        AreaStat(source="s", model="m", area="mid", n=10, errors=5),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["s"])
    assert [a.area for a, _, _ in table.ranked()] == ["high", "mid", "low"]


def test_empty_table_is_honest() -> None:
    table = BedTable(min_n=5, areas=[], sources=[])
    assert table.weights() == {}
    assert table.ranked() == []
    assert "measured" in table.to_markdown().lower() or table.to_markdown()


def test_build_table_skips_missing_files(tmp_path: Path, principle_payload: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    table = build_table([missing, principle_payload], min_n=5)
    assert table.sources == ["principle_bench_results.json"]
    assert len(table.areas) == 4


def test_build_table_reports_error_criterion_explicitly(principle_payload: Path) -> None:
    table = build_table([principle_payload], min_n=5)
    payload = table.to_dict()
    assert payload["error_criterion"]["principle_bench"].startswith("sample.ok is False")
    assert payload["error_criterion"]["vtb"].endswith("THEATER")
    assert payload["n_areas_weighted"] == 4
    assert payload["n_areas_insufficient"] == 0


# ── saturation / aggregation (the defect measured 2026-09-15) ────────────────


def test_saturated_source_is_detected_and_excluded_from_weights() -> None:
    """All-100% source reports a state but ranks nothing: no weight for it."""
    areas = [
        AreaStat(source="saturated", model="m/netelpro", area=f"f{i}", n=5, errors=5)
        for i in range(3)
    ] + [
        AreaStat(source="discriminative", model="base", area="weak", n=10, errors=3),
        AreaStat(source="discriminative", model="base", area="ok", n=10, errors=0),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["saturated", "discriminative"])
    sat = table.saturation()
    assert sat["saturated"]["saturated"] is True
    assert sat["saturated"]["carries_ranking_information"] is False
    assert sat["discriminative"]["saturated"] is False
    assert sat["discriminative"]["carries_ranking_information"] is True

    weights = table.weights_by_area()
    assert all("saturated" not in k for k in weights)
    assert weights["discriminative|weak"] == pytest.approx(1.0)


def test_all_saturated_sources_yield_empty_weight_vector() -> None:
    """The honest answer when nothing ranks: no weights, not a fake uniform spread."""
    areas = [
        AreaStat(source="a", model="m", area="x", n=5, errors=5),
        AreaStat(source="b", model="m", area="y", n=5, errors=0),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["a", "b"])
    assert table.saturation()["a"]["saturated"] is True
    assert table.saturation()["b"]["saturated"] is True
    assert table.weights_by_area() == {}
    assert table.ranked_by_area() == []
    assert "nothing to direct training with" in table.to_markdown()


def test_aggregate_by_area_collapses_conditions() -> None:
    areas = [
        AreaStat(source="s", model="m1/netelpro", area="fam", n=5, errors=4, classes={"truncado": 4}),
        AreaStat(source="s", model="m2/json", area="fam", n=5, errors=2, classes={"estructura": 2}),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["s"])
    agg = table.aggregate_by_area()
    assert len(agg) == 1
    assert agg[0].n == 10
    assert agg[0].errors == 6
    assert agg[0].error_rate == pytest.approx(0.6)
    assert agg[0].classes == {"truncado": 4, "estructura": 2}
    assert "2 conditions" in agg[0].model


def test_aggregation_prevents_saturated_cells_from_burying_signal() -> None:
    """Regression: the 2026-09-15 defect. 96 saturated cells must not bury VTB."""
    areas = [
        AreaStat(source="bench", model=f"m{i}/{fmt}", area=f"f{i}", n=5, errors=5)
        for i in range(20)
        for fmt in ("netelpro", "json")
    ] + [
        AreaStat(source="vtb", model="base", area="weak", n=10, errors=3),
        AreaStat(source="vtb", model="base", area="strong", n=10, errors=0),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["bench", "vtb"])
    # Cell-level weights would hand 40/41 of the mass to the saturated source.
    cell_weights = table.weights()
    bench_mass = sum(w for k, w in cell_weights.items() if k.startswith("bench|"))
    assert bench_mass > 0.9
    # Aggregated + saturation-filtered weights hand ALL of it to the ranking source.
    area_weights = table.weights_by_area()
    assert all(k.startswith("vtb|") for k in area_weights)
    assert sum(area_weights.values()) == pytest.approx(1.0)


# ── against the real artifacts on disk (skipped if absent) ───────────────────


# ── evidence-aware weighting (the defect measured 2026-09-15) ────────────────


def test_wilson_lower_bound_is_below_the_point_estimate() -> None:
    # 3/20 = 15.0% but the data only proves ~5.2%; 6/40 = 15.0% proves ~7.1%.
    assert wilson_lower_bound(3, 20) == pytest.approx(0.0524, abs=5e-4)
    assert wilson_lower_bound(6, 40) == pytest.approx(0.0706, abs=5e-4)
    assert wilson_lower_bound(3, 20) < wilson_lower_bound(6, 40)
    # A bigger sample with the same rate proves more: 30/200 vs 3/20.
    assert wilson_lower_bound(30, 200) > wilson_lower_bound(3, 20)
    # Degenerate inputs must not raise or invent evidence.
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(0, 10) == 0.0
    assert wilson_lower_bound(10, 10) < 1.0


def test_equal_rates_from_unequal_n_do_not_get_equal_weight() -> None:
    """The measured defect: 15% of 20 and 15% of 40 are not the same evidence."""
    areas = [
        AreaStat(source="s", model="m", area="small", n=20, errors=3),
        AreaStat(source="s", model="m", area="large", n=40, errors=6),
        AreaStat(source="s", model="m", area="clean", n=20, errors=0),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["s"])
    point = table.weights_by_area()
    # Point estimate splits the mass evenly between the two 15% areas.
    assert point["s|small"] == pytest.approx(point["s|large"])
    conservative = table.weights_by_area(conservative=True)
    # The larger sample keeps the bigger share once noise is discounted.
    assert conservative["s|large"] > conservative["s|small"]
    assert conservative["s|clean"] == 0.0
    assert sum(conservative.values()) == pytest.approx(1.0)


def test_rank_stability_flags_a_rank_bought_by_noise() -> None:
    """Equal point estimates, different evidence: the tie-break flips."""
    areas = [
        AreaStat(source="s", model="m", area="lucky_small", n=20, errors=3),
        AreaStat(source="s", model="m", area="solid_large", n=200, errors=28),
        AreaStat(source="s", model="m", area="clean", n=200, errors=0),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["s"])
    rows = {row["key"]: row for row in table.rank_stability()}

    # The small sample ranks FIRST on the point estimate (15.0% beats 14.0%).
    assert rows["s|lucky_small"]["rank_point_estimate"] == 1
    assert rows["s|solid_large"]["rank_point_estimate"] == 2
    assert rows["s|lucky_small"]["error_rate"] > rows["s|solid_large"]["error_rate"]

    # Under the bound the 200-sample area takes the top rank: the small
    # sample's lead was bought by sampling noise, not by evidence.
    assert rows["s|solid_large"]["rank_conservative"] == 1
    assert rows["s|lucky_small"]["rank_conservative"] == 2
    assert rows["s|solid_large"]["rank_supported_by_n"] is True
    assert rows["s|solid_large"]["rank_understated"] is True
    assert rows["s|solid_large"]["rank_inflated_by_noise"] is False
    assert rows["s|lucky_small"]["rank_supported_by_n"] is False
    assert rows["s|lucky_small"]["rank_inflated_by_noise"] is True
    assert rows["s|lucky_small"]["rank_understated"] is False
    assert rows["s|lucky_small"]["proven_weak"] is True
    assert rows["s|solid_large"]["lower_bound"] > rows["s|lucky_small"]["lower_bound"]
    assert rows["s|clean"]["proven_weak"] is False
    assert rows["s|clean"]["rank_supported_by_n"] is True
    assert rows["s|clean"]["rank_inflated_by_noise"] is False
    assert rows["s|clean"]["rank_understated"] is False


def test_zero_error_area_is_not_proven_weak() -> None:
    areas = [
        AreaStat(source="s", model="m", area="weak", n=20, errors=4),
        AreaStat(source="s", model="m", area="clean", n=20, errors=0),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["s"])
    rows = {row["key"]: row for row in table.rank_stability()}
    assert rows["s|clean"]["proven_weak"] is False
    assert rows["s|clean"]["lower_bound"] == 0.0
    assert rows["s|weak"]["proven_weak"] is True
    assert table.to_dict()["n_areas_proven_weak"] == 1


def test_conservative_vector_is_honest_when_nothing_is_proven() -> None:
    """All-clean data: no area has a proven error, so no weight anywhere."""
    areas = [AreaStat(source="s", model="m", area=f"a{i}", n=10, errors=0) for i in range(3)]
    table = BedTable(min_n=5, areas=areas, sources=["s"])
    # Every area is at 0% -> the source is saturated -> no vector at all,
    # not a uniform spread invented to fill the table.
    assert table.weights_by_area(conservative=True) == {}

    # And when the source DOES rank, the clean area still gets exactly 0.0.
    mixed = BedTable(
        min_n=5,
        areas=[
            AreaStat(source="s", model="m", area="weak", n=20, errors=4),
            AreaStat(source="s", model="m", area="clean", n=20, errors=0),
        ],
        sources=["s"],
    )
    conservative = mixed.weights_by_area(conservative=True)
    assert conservative["s|clean"] == 0.0
    assert conservative["s|weak"] == pytest.approx(1.0)


def test_markdown_reports_rank_stability_and_conservative_vector() -> None:
    areas = [
        AreaStat(source="s", model="m", area="small", n=20, errors=3),
        AreaStat(source="s", model="m", area="large", n=200, errors=28),
        AreaStat(source="s", model="m", area="clean", n=200, errors=0),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["s"])
    md = table.to_markdown()
    assert "Rank stability" in md
    assert "Wilson 95% lower bound" in md
    assert "Conservative vector" in md
    assert "INFLATED BY NOISE" in md
    assert "understated — more data would raise it" in md
    assert "confirmed by n" in md


def test_area_stat_dict_carries_the_lower_bound() -> None:
    stat = AreaStat(source="s", model="m", area="a", n=20, errors=3)
    payload = stat.to_dict()
    assert payload["error_rate"] == pytest.approx(0.15)
    assert payload["lower_bound"] == pytest.approx(0.0524, abs=5e-4)


@pytest.mark.skipif(
    not (BENCH_DIR / "area_error_table.json").is_file(),
    reason="bed artifact not built in this checkout",
)
def test_real_bed_artifact_exposes_conservative_weights() -> None:
    payload = json.loads((BENCH_DIR / "area_error_table.json").read_text(encoding="utf-8"))
    assert "weights_by_area_conservative" in payload
    assert payload["rank_stability"]
    # Every stability row must state whether its rank is supported.
    assert all("rank_supported_by_n" in row for row in payload["rank_stability"])


# ── failure-mode profile: the signal a saturated rate hides ──────────────────


def test_class_profile_reports_shares_that_sum_to_one() -> None:
    areas = [
        AreaStat(
            source="bench",
            model="m/netelpro",
            area="fam_a",
            n=10,
            errors=10,
            classes={"truncado": 8, "estructura": 2},
        ),
        AreaStat(
            source="bench",
            model="m/json",
            area="fam_b",
            n=10,
            errors=10,
            classes={"sin_bloque": 6, "parseo": 4},
        ),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["bench"])
    profiles = {row["key"]: row for row in table.class_profile_by_area()}
    assert sum(profiles["bench|fam_a"]["shares"].values()) == pytest.approx(1.0)
    assert profiles["bench|fam_a"]["dominant_class"] == "truncado"
    assert profiles["bench|fam_a"]["dominant_share"] == pytest.approx(0.8)
    assert profiles["bench|fam_b"]["dominant_class"] == "sin_bloque"


def test_saturated_source_can_still_rank_its_failure_modes() -> None:
    """The measured case: 100% everywhere by rate, different modes underneath."""
    areas = [
        AreaStat(
            source="bench",
            model="m/netelpro",
            area="fam_a",
            n=10,
            errors=10,
            classes={"truncado": 9, "estructura": 1},
        ),
        AreaStat(
            source="bench",
            model="m/json",
            area="fam_b",
            n=10,
            errors=10,
            classes={"sin_bloque": 9, "estructura": 1},
        ),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["bench"])
    # By rate the source is saturated: no ranking, no weight.
    assert table.saturation()["bench"]["saturated"] is True
    assert table.weights_by_area() == {}
    # By failure mode it ranks: two distinct dominant classes.
    modes = table.failure_mode_ranking()["bench"]
    assert modes["ranks_failure_modes"] is True
    assert modes["dominant_classes"] == ["sin_bloque", "truncado"]
    assert modes["worst_concentration"]["class"] == "truncado"
    assert modes["worst_concentration"]["share"] == pytest.approx(0.9)


def test_failure_modes_never_enter_the_weight_vector() -> None:
    """Direction is not severity: a saturated source gets 0 weight regardless."""
    areas = [
        AreaStat(
            source="bench",
            model="m",
            area=f"fam{i}",
            n=10,
            errors=10,
            classes={"truncado": 10} if i == 0 else {"sin_bloque": 10},
        )
        for i in range(2)
    ] + [
        AreaStat(source="vtb", model="base", area="weak", n=20, errors=4),
        AreaStat(source="vtb", model="base", area="strong", n=20, errors=0),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["bench", "vtb"])
    weights = table.weights_by_area()
    assert all(k.startswith("vtb|") for k in weights)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_areas_without_classes_are_absent_from_the_profile() -> None:
    areas = [AreaStat(source="s", model="m", area="clean", n=10, errors=0)]
    table = BedTable(min_n=5, areas=areas, sources=["s"])
    assert table.class_profile_by_area() == []
    assert table.failure_mode_ranking() == {}


def test_markdown_shows_the_failure_mode_table_for_saturated_sources() -> None:
    areas = [
        AreaStat(
            source="bench",
            model="m",
            area="fam_a",
            n=10,
            errors=10,
            classes={"truncado": 9, "estructura": 1},
        ),
        AreaStat(
            source="bench",
            model="m",
            area="fam_b",
            n=10,
            errors=10,
            classes={"sin_bloque": 9, "estructura": 1},
        ),
    ]
    table = BedTable(min_n=5, areas=areas, sources=["bench"])
    md = table.to_markdown()
    assert "What a saturated rate hides" in md
    assert "failure-mode profile" in md
    assert "truncado" in md and "sin_bloque" in md


@pytest.mark.skipif(
    not (BENCH_DIR / "principle_bench_results.json").is_file(),
    reason="real principle bench results not present in this checkout",
)
def test_real_principle_bench_areas_are_the_12_fresh_families() -> None:
    table = build_table([BENCH_DIR / "principle_bench_results.json"], min_n=5)
    families = {a.area for a in table.areas}
    assert len(families) == 12
    assert "chatelier_ratelimit" in families
    # 4 models x 2 formats x 12 families, all with 5 samples each.
    assert len(table.areas) == 96
    assert all(a.n == 5 for a in table.areas)


@pytest.mark.skipif(
    not all(p.is_file() for p in default_paths()),
    reason="real benchmark artifacts not present in this checkout",
)
def test_real_default_build_is_consistent() -> None:
    table = build_table(default_paths(), min_n=5)
    payload = table.to_dict()
    assert payload["n_areas_measured"] == len(table.areas)
    assert payload["n_areas_weighted"] + payload["n_areas_insufficient"] == len(table.areas)
    weights = table.weights()
    assert all(0.0 <= w <= 1.0 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)
