#!/usr/bin/env python3
"""Statistical bed (Epistemic Gate, consumer B) — measured error rates per area.

Spec: docs/EPISTEMIC_GATE_SPEC.md v0.1, section 1 (consumer B) and section 7
(honesty constraints). This module is the *first food* of the shared map: it
turns the benchmark result files that already exist on disk into per-area error
rates plus a training-weight vector, so the next run is directed at the areas
the arbiter actually measured as weak.

Design constraints (non-negotiable, spec section 7):

1. **No fabrication.** An area with fewer than ``min_n`` samples is reported as
   ``insufficient_data`` and is EXCLUDED from the weight vector. It never gets a
   fabricated 0.0 error rate (a 0.0 from no data is a lie by omission).
2. **Fail-closed per source.** The error criterion is declared explicitly per
   source, never inferred:
     - principle bench: a sample is an error iff the real compiler rejected it
       (``ok is False``); the error class is the grader's ``err`` label.
     - VTB tables: a sample is an error iff the verdict is ``THEATER``
       (verification theater is the failure mode the house cares about).
3. **Auditable.** Every area carries source, model/condition, n, error count and
   the error-class breakdown. The weight vector is a pure function of those
   numbers; nothing is hand-tuned.

Usage:
    python benchmarks/area_error_table.py --build
    python benchmarks/area_error_table.py --print
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmarks"

DEFAULT_PRINCIPLE_BENCH = BENCH_DIR / "principle_bench_results.json"
DEFAULT_VTB_QWEN = BENCH_DIR / "vtb_qwen_local_benchmark_results.json"
DEFAULT_VTB_LFM = BENCH_DIR / "vtb_lfm_benchmark_results.json"
DEFAULT_VTB_OOD_V2 = BENCH_DIR / "vtb_ood_benchmark_results_v2.json"

OUT_JSON = BENCH_DIR / "area_error_table.json"
OUT_MD = BENCH_DIR / "area_error_table.md"

#: Minimum samples per area for the area to enter the weight vector.
DEFAULT_MIN_N = 5

#: VTB verdict treated as the error of record (the house failure mode).
VTB_ERROR_STATUS = "THEATER"

#: z for the 95% Wilson interval behind the conservative weight vector.
WILSON_Z = 1.96


def wilson_lower_bound(errors: int, n: int, z: float = WILSON_Z) -> float:
    """Lower bound of the Wilson score interval for ``errors / n`` (95% by default).

    Why the bed needs this (measured 2026-09-15): two areas can show the SAME
    rate from different sample sizes — 15% of 20 and 15% of 40 — and the point
    estimate ranks them as equals. It is not the same evidence. The lower bound
    is the rate the data still supports once sampling noise is discounted, so a
    small noisy sample cannot buy weight with a lucky rate.
    """
    if n <= 0:
        return 0.0
    p = errors / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, centre - half)


@dataclass(frozen=True)
class AreaStat:
    """Measured error rate of one area, for one model/condition, from one source."""

    source: str
    model: str
    area: str
    n: int
    errors: int
    classes: Dict[str, int] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        return (self.errors / self.n) if self.n else 0.0

    @property
    def lower_bound(self) -> float:
        """Conservative error rate: Wilson 95% lower bound (0.0 when n == 0)."""
        return wilson_lower_bound(self.errors, self.n)

    @property
    def key(self) -> str:
        return f"{self.source}|{self.model}|{self.area}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "model": self.model,
            "area": self.area,
            "n": self.n,
            "errors": self.errors,
            "error_rate": round(self.error_rate, 6),
            "lower_bound": round(self.lower_bound, 6),
            "classes": dict(sorted(self.classes.items())),
        }


@dataclass
class BedTable:
    """The statistical bed: measured areas + the weight vector derived from them."""

    min_n: int
    areas: List[AreaStat]
    sources: List[str]
    generated_at: str = ""

    # ── selection ────────────────────────────────────────────────────────────

    def sufficient(self) -> List[AreaStat]:
        """Areas with enough samples to carry a weight (spec 7: no fabrication)."""
        return [a for a in self.areas if a.n >= self.min_n]

    def insufficient(self) -> List[AreaStat]:
        """Areas measured but with too few samples to weigh. Reported, not weighted."""
        return [a for a in self.areas if a.n < self.min_n]

    # ── aggregation ──────────────────────────────────────────────────────────

    def aggregate_by_area(self) -> List[AreaStat]:
        """Collapse (model, format) cells into one stat per (source, area).

        The unit that can direct training is the AREA, not the area x model x
        format cell. Measured 2026-09-15: the cell-level view let 96 saturated
        principle-bench cells (all 100%) bury the 24 discriminative VTB cells.
        """
        buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for a in self.areas:
            bucket = buckets.setdefault(
                (a.source, a.area), {"n": 0, "errors": 0, "classes": {}, "models": 0}
            )
            bucket["n"] += a.n
            bucket["errors"] += a.errors
            bucket["models"] += 1
            for label, count in a.classes.items():
                bucket["classes"][label] = bucket["classes"].get(label, 0) + count
        return [
            AreaStat(
                source=source,
                model=f"({bucket['models']} conditions)",
                area=area,
                n=bucket["n"],
                errors=bucket["errors"],
                classes=bucket["classes"],
            )
            for (source, area), bucket in sorted(buckets.items())
        ]

    def saturation(self) -> Dict[str, Dict[str, Any]]:
        """Per-source spread of error rates — does this source rank anything?

        A source whose areas are ALL at 0% or ALL at 100% is saturated: it
        reports a failure state but carries zero ranking information, so it
        cannot direct training. Saying that out loud is the honest move; the
        alternative is a weight vector that looks quantitative and means nothing.
        """
        report: Dict[str, Dict[str, Any]] = {}
        for stat in self.aggregate_by_area():
            if stat.n < self.min_n:
                continue
            entry = report.setdefault(
                stat.source, {"rates": [], "areas": 0, "flat": set()}
            )
            entry["rates"].append(stat.error_rate)
            entry["areas"] += 1
            entry["flat"].add(round(stat.error_rate, 6))
        out: Dict[str, Dict[str, Any]] = {}
        for source, entry in sorted(report.items()):
            rates: List[float] = entry["rates"]
            flat = sorted(entry["flat"])
            out[source] = {
                "areas": entry["areas"],
                "min_rate": round(min(rates), 6),
                "max_rate": round(max(rates), 6),
                "distinct_rates": flat,
                "saturated": len(flat) == 1,
                "carries_ranking_information": len(flat) > 1,
            }
        return out

    def discriminative_areas(self) -> List[AreaStat]:
        """Aggregated areas from sources that actually rank (not saturated)."""
        sat = self.saturation()
        return [
            a
            for a in self.aggregate_by_area()
            if a.n >= self.min_n and sat.get(a.source, {}).get("carries_ranking_information")
        ]

    def weights_by_area(self, *, conservative: bool = False) -> Dict[str, float]:
        """Normalized training weight per discriminative area (aggregated).

        Built ONLY from non-saturated sources: a saturated source contributes
        no ranking, so letting it into the vector would be arithmetic theater.
        Keys are ``source|area`` (the aggregated unit, not the cell).

        ``conservative=True`` weights by the Wilson 95% LOWER BOUND instead of
        the point estimate. An area whose lower bound is 0.0 — no measured
        error, or too little data to prove one — gets 0.0, never a share
        fabricated out of a small sample.
        """
        areas = self.discriminative_areas()
        magnitude = (lambda a: a.lower_bound) if conservative else (lambda a: a.error_rate)
        total = sum(magnitude(a) for a in areas)
        if total <= 0.0:
            return {f"{a.source}|{a.area}": 0.0 for a in areas}
        return {f"{a.source}|{a.area}": magnitude(a) / total for a in areas}

    def ranked_by_area(
        self, *, conservative: bool = False
    ) -> List[Tuple[AreaStat, float, int]]:
        """(area, weight, rank) over discriminative areas, descending weight."""
        weights = self.weights_by_area(conservative=conservative)
        areas = sorted(
            self.discriminative_areas(),
            key=lambda a: (-weights[f"{a.source}|{a.area}"], a.area),
        )
        return [(a, weights[f"{a.source}|{a.area}"], i + 1) for i, a in enumerate(areas)]

    def rank_stability(self) -> List[Dict[str, Any]]:
        """Does the sample size support each rank the point estimate produced?

        Compares the ranking by measured rate against the ranking by Wilson
        lower bound. Three honest verdicts, not two:

        - ``rank_confirmed``: the bound keeps the area at its rank (or better),
          so the data does not contradict the point estimate.
        - ``rank_inflated_by_noise``: the bound DEMOTES the area. Its position
          was bought by a small sample, not by evidence.
        - ``rank_understated``: the bound PROMOTES the area. The point estimate
          was too shy; more data would raise it.

        An earlier version called any movement "noise", which libelled the
        understated areas. Measured 2026-09-15 against the real bed.
        """
        conservative_rank = {
            f"{a.source}|{a.area}": rank for a, _, rank in self.ranked_by_area(conservative=True)
        }
        rows: List[Dict[str, Any]] = []
        for area, weight, rank in self.ranked_by_area():
            key = f"{area.source}|{area.area}"
            conservative = conservative_rank.get(key, rank)
            rows.append(
                {
                    "key": key,
                    "n": area.n,
                    "errors": area.errors,
                    "error_rate": round(area.error_rate, 6),
                    "lower_bound": round(area.lower_bound, 6),
                    "weight": round(weight, 6),
                    "rank_point_estimate": rank,
                    "rank_conservative": conservative,
                    "rank_supported_by_n": conservative <= rank,
                    "rank_inflated_by_noise": conservative > rank,
                    "rank_understated": conservative < rank,
                    "proven_weak": area.lower_bound > 0.0,
                }
            )
        return rows

    def class_profile_by_area(self) -> List[Dict[str, Any]]:
        """Failure-MODE profile per aggregated area — the signal a saturated rate hides.

        Measured 2026-09-15: every principle-bench family fails at exactly 100%,
        so ``saturation()`` correctly refuses to rank them and hands them zero
        weight. But the *composition* of the failures is not flat: one family
        fails 38% by truncation, another 25% by missing blocks. The headline rate
        says "broken"; the profile says "broken HOW", and that is what decides
        which data to generate next.

        This does NOT feed the weight vector — severity still comes only from
        measured rates (spec 7: no fabrication). It is reported as direction.
        """
        rows: List[Dict[str, Any]] = []
        for area in self.aggregate_by_area():
            if area.n < self.min_n:
                continue
            total = sum(area.classes.values())
            if not total:
                continue
            shares = {k: v / total for k, v in area.classes.items()}
            dominant = max(sorted(shares.items()), key=lambda kv: kv[1])[0]
            rows.append(
                {
                    "key": f"{area.source}|{area.area}",
                    "source": area.source,
                    "n": area.n,
                    "errors": area.errors,
                    "error_rate": round(area.error_rate, 6),
                    "shares": {k: round(v, 6) for k, v in sorted(shares.items())},
                    "dominant_class": dominant,
                    "dominant_share": round(shares[dominant], 6),
                }
            )
        return rows

    def failure_mode_ranking(self) -> Dict[str, Dict[str, Any]]:
        """Per source: can its failure MODES rank, even when its rate cannot?"""
        by_source: Dict[str, List[Dict[str, Any]]] = {}
        for row in self.class_profile_by_area():
            by_source.setdefault(row["source"], []).append(row)

        out: Dict[str, Dict[str, Any]] = {}
        for source, rows in sorted(by_source.items()):
            dominants = sorted({row["dominant_class"] for row in rows})
            # Explicit ordering: highest concentration first, ties by key.
            # (``max`` on a tuple would silently tie-break to the LARGEST key.)
            worst = sorted(rows, key=lambda r: (-r["dominant_share"], r["key"]))[0]
            out[source] = {
                "areas": len(rows),
                "dominant_classes": dominants,
                "ranks_failure_modes": len(dominants) > 1,
                "worst_concentration": {
                    "key": worst["key"],
                    "class": worst["dominant_class"],
                    "share": worst["dominant_share"],
                },
            }
        return out

    # ── weight vector (consumer B) ───────────────────────────────────────────

    def weights(self) -> Dict[str, float]:
        """Normalized training weight per sufficient area.

        ``w_i = error_rate_i / sum(error_rate_j)`` over the sufficient areas.
        If no area shows a measured error, every weight is 0.0 — the honest
        answer is "nothing measured as weak", not a uniform spread.
        """
        suff = self.sufficient()
        total = sum(a.error_rate for a in suff)
        if total <= 0.0:
            return {a.key: 0.0 for a in suff}
        return {a.key: a.error_rate / total for a in suff}

    def ranked(self) -> List[Tuple[AreaStat, float, int]]:
        """(area, weight, rank) sorted by descending weight, ties by area key."""
        weights = self.weights()
        suff = sorted(self.sufficient(), key=lambda a: (-weights[a.key], a.key))
        return [(a, weights[a.key], i + 1) for i, a in enumerate(suff)]

    # ── serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        weights = self.weights()
        return {
            "generated_at": self.generated_at,
            "spec": "straylight/docs/EPISTEMIC_GATE_SPEC.md v0.1 — Phase E (consumer B)",
            "min_n": self.min_n,
            "sources": self.sources,
            "error_criterion": {
                "principle_bench": "sample.ok is False (real compiler rejected it)",
                "vtb": f"verdict == {VTB_ERROR_STATUS}",
            },
            "n_areas_measured": len(self.areas),
            "n_areas_weighted": len(self.sufficient()),
            "n_areas_insufficient": len(self.insufficient()),
            "saturation": self.saturation(),
            "n_discriminative_areas": len(self.discriminative_areas()),
            "weights_by_area": {
                k: round(v, 6) for k, v in sorted(self.weights_by_area().items())
            },
            "weights_by_area_conservative": {
                k: round(v, 6)
                for k, v in sorted(self.weights_by_area(conservative=True).items())
            },
            "n_areas_proven_weak": sum(1 for row in self.rank_stability() if row["proven_weak"]),
            "rank_stability": self.rank_stability(),
            "failure_mode_ranking": self.failure_mode_ranking(),
            "class_profile_by_area": self.class_profile_by_area(),
            "areas": [a.to_dict() for a in self.areas],
            "areas_aggregated": [a.to_dict() for a in self.aggregate_by_area()],
            "weights": {k: round(v, 6) for k, v in sorted(weights.items())},
        }

    def to_markdown(self) -> str:
        lines: List[str] = [
            "# Statistical bed — measured error rate per area (Epistemic Gate, consumer B)",
            "",
            f"* Generated: {self.generated_at} · min_n (weight threshold): {self.min_n}",
            "* Error criterion is declared per source, never inferred:",
            (
                "  principle bench = the real compiler rejected the generation;"
                f" VTB = verdict `{VTB_ERROR_STATUS}`."
            ),
            (
                "* Areas under `min_n` samples are reported but get NO weight"
                " (a 0.0 without data is a lie by omission)."
            ),
            "",
        ]

        stability = self.rank_stability()
        if stability:
            supported = sum(1 for row in stability if row["rank_supported_by_n"])
            inflated = sum(1 for row in stability if row["rank_inflated_by_noise"])
            understated = sum(1 for row in stability if row["rank_understated"])
            proven = sum(1 for row in stability if row["proven_weak"])
            lines.append("## Rank stability — is each rank paid for by the sample size?")
            lines.append("")
            lines.append(
                "* Same rate from different `n` is not the same evidence: ranking by the point"
                " estimate is compared against ranking by the Wilson 95% lower bound."
            )
            lines.append(
                f"* Ranks the data does not contradict: **{supported}/{len(stability)}** ·"
                f" inflated by noise: **{inflated}** · understated (bound promotes them):"
                f" **{understated}** · areas with a PROVEN error rate: **{proven}**"
            )
            lines.append("")
            lines.append(
                "| Area | n | Errors | Rate | Wilson 95% lower bound | Rank (point) | Rank (conservative) | Verdict |"
            )
            lines.append("|---|---|---|---|---|---|---|---|")
            for row in stability:
                if row["rank_inflated_by_noise"]:
                    verdict = "INFLATED BY NOISE"
                elif row["rank_understated"]:
                    verdict = "understated — more data would raise it"
                else:
                    verdict = "confirmed by n"
                lines.append(
                    f"| {row['key']} | {row['n']} | {row['errors']} | "
                    f"{row['error_rate'] * 100:.1f}% | {row['lower_bound'] * 100:.1f}% | "
                    f"{row['rank_point_estimate']} | {row['rank_conservative']} | "
                    f"{verdict} |"
                )
            lines.append("")

        lines.append("## Saturation check — does each source rank anything?")
        lines.append("")
        lines.append("| Source | Areas | Min rate | Max rate | Distinct rates | Verdict |")
        lines.append("|---|---|---|---|---|---|")
        for source, info in self.saturation().items():
            verdict = (
                "SATURATED — reports a state, ranks nothing, excluded from weights"
                if info["saturated"]
                else "ranks areas (usable)"
            )
            lines.append(
                f"| {source} | {info['areas']} | {info['min_rate'] * 100:.1f}% | "
                f"{info['max_rate'] * 100:.1f}% | {len(info['distinct_rates'])} | {verdict} |"
            )
        lines.append("")

        saturation = self.saturation()
        saturated_sources = {
            source
            for source, info in saturation.items()
            if not info["carries_ranking_information"]
        }
        profiles = [
            row
            for row in self.class_profile_by_area()
            if row["source"] in saturated_sources
        ]
        if profiles:
            modes = self.failure_mode_ranking()
            lines.append("## What a saturated rate hides — failure-mode profile")
            lines.append("")
            lines.append(
                "* A source that cannot rank by rate is not necessarily flat: the"
                " COMPOSITION of its failures may still rank. Reported as direction,"
                " never as weight (severity comes only from measured rates)."
            )
            lines.append("")
            lines.append(
                "| Source | Areas | Distinct dominant classes | Ranks failure modes? | Most concentrated |"
            )
            lines.append("|---|---|---|---|---|")
            for source in sorted(saturated_sources):
                info = modes.get(source)
                if not info:
                    continue
                worst = info["worst_concentration"]
                lines.append(
                    f"| {source} | {info['areas']} | {len(info['dominant_classes'])} | "
                    f"{'yes' if info['ranks_failure_modes'] else 'no'} | "
                    f"{worst['key']} → {worst['class']} ({worst['share'] * 100:.0f}%) |"
                )
            lines.append("")

            classes = sorted({c for row in profiles for c in row["shares"]})
            lines.append("| Area | n | " + " | ".join(classes) + " | Dominant |")
            lines.append("|---" * (len(classes) + 3) + "|")
            for row in sorted(profiles, key=lambda r: r["key"]):
                cells = " | ".join(
                    f"{row['shares'].get(c, 0.0) * 100:.0f}%" for c in classes
                )
                lines.append(
                    f"| {row['key']} | {row['n']} | {cells} | "
                    f"{row['dominant_class']} ({row['dominant_share'] * 100:.0f}%) |"
                )
            lines.append("")

        ranked = self.ranked_by_area()
        lines.append("## Training weight vector (discriminative areas only, aggregated)")
        lines.append("")
        if not ranked:
            lines.append("_No source carried ranking information — nothing to direct training with._")
            lines.append("")
        else:
            lines.append("| # | Source | Area | n | Errors | Error rate | Weight |")
            lines.append("|---|---|---|---|---|---|---|")
            for area, weight, rank in ranked:
                lines.append(
                    f"| {rank} | {area.source} | {area.area} | {area.n} | {area.errors} | "
                    f"{area.error_rate * 100:.1f}% | {weight * 100:.1f}% |"
                )
            lines.append("")

            conservative = self.weights_by_area(conservative=True)
            lines.append("### Conservative vector (Wilson 95% lower bound, noise discounted)")
            lines.append("")
            lines.append("| # | Area | n | Errors | Rate | Weight (conservative) |")
            lines.append("|---|---|---|---|---|---|")
            for area, _, rank in self.ranked_by_area(conservative=True):
                key = f"{area.source}|{area.area}"
                lines.append(
                    f"| {rank} | {key} | {area.n} | {area.errors} | "
                    f"{area.error_rate * 100:.1f}% | {conservative[key] * 100:.1f}% |"
                )
            lines.append("")

        lines.append("## All weighted areas (per source x model x format cell)")
        lines.append("")
        lines.append("| # | Source | Model/condition | Area | n | Errors | Error rate | Weight |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for area, weight, rank in self.ranked():
            lines.append(
                f"| {rank} | {area.source} | {area.model} | {area.area} | {area.n} | "
                f"{area.errors} | {area.error_rate * 100:.1f}% | {weight * 100:.1f}% |"
            )
        lines.append("")

        insuff = self.insufficient()
        if insuff:
            lines.append("## Measured but insufficient (excluded from the weights)")
            lines.append("")
            lines.append("| Source | Model/condition | Area | n | Errors |")
            lines.append("|---|---|---|---|---|")
            for a in sorted(insuff, key=lambda a: a.key):
                lines.append(f"| {a.source} | {a.model} | {a.area} | {a.n} | {a.errors} |")
            lines.append("")

        lines.append("## Error classes (where the failures come from)")
        lines.append("")
        lines.append("| Source | Model/condition | Area | Classes |")
        lines.append("|---|---|---|---|")
        for a in sorted(self.areas, key=lambda a: a.key):
            if not a.classes:
                continue
            rendered = ", ".join(f"{k}: {v}" for k, v in sorted(a.classes.items()))
            lines.append(f"| {a.source} | {a.model} | {a.area} | {rendered} |")
        lines.append("")
        return "\n".join(lines)


# ── loaders: one per source, each with its error criterion spelled out ───────


def load_principle_bench(path: Path) -> List[AreaStat]:
    """Per-family error rates from principle_bench_results.json part B.

    Error iff the grader rejected the sample; class = the grader's ``err`` label
    (``truncado`` / ``sin_casos`` / ``semantica`` / ``compilacion`` / ``sin_bloque``).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = data.get("part_b", {}).get("cells", {})
    stats: List[AreaStat] = []
    for cell_key, cell in sorted(cells.items()):
        model = cell_key.split("|", 1)[0]
        family = cell.get("family") or cell_key.split("|", 1)[-1].split("#", 1)[0]
        for fmt in ("netelpro", "json"):
            block = cell.get(fmt) or {}
            samples = block.get("samples") or []
            classes: Dict[str, int] = {}
            errors = 0
            for sample in samples:
                if sample.get("ok") is True:
                    continue
                errors += 1
                label = sample.get("err") or "sin_clase"
                classes[label] = classes.get(label, 0) + 1
            stats.append(
                AreaStat(
                    source="principle_bench",
                    model=f"{model}/{fmt}",
                    area=family,
                    n=len(samples),
                    errors=errors,
                    classes=classes,
                )
            )
    return stats


def load_vtb_case_table(path: Path, source: str) -> List[AreaStat]:
    """Per-category theater rates from a VTB case_comparisons table.

    Two models per file (base / aligned); error iff the scorer's verdict is
    ``THEATER``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("case_comparisons") or []
    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for case in cases:
        category = case.get("category") or "sin_categoria"
        for role, status_key in (("base", "base_status"), ("aligned", "aligned_status")):
            status = case.get(status_key)
            if status is None:
                continue
            bucket = buckets.setdefault((role, category), {"n": 0, "errors": 0, "classes": {}})
            bucket["n"] += 1
            if status == VTB_ERROR_STATUS:
                bucket["errors"] += 1
                bucket["classes"]["theater"] = bucket["classes"].get("theater", 0) + 1
    return [
        AreaStat(
            source=source,
            model=role,
            area=category,
            n=bucket["n"],
            errors=bucket["errors"],
            classes=bucket["classes"],
        )
        for (role, category), bucket in sorted(buckets.items())
    ]


def load_vtb_ood_v2(path: Path) -> List[AreaStat]:
    """Per-category theater rates from the 2x2 prompt-vs-training OOD experiment.

    One condition per model role (base / base_sys / aligned / aligned_sys);
    error iff the scorer's verdict is ``THEATER``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("case_results") or []
    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for case in cases:
        category = case.get("category") or "sin_categoria"
        for role, payload in (case.get("responses") or {}).items():
            status = payload.get("status_v2_scorer", payload.get("status"))
            if status is None:
                continue
            bucket = buckets.setdefault((role, category), {"n": 0, "errors": 0, "classes": {}})
            bucket["n"] += 1
            if status == VTB_ERROR_STATUS:
                bucket["errors"] += 1
                bucket["classes"]["theater"] = bucket["classes"].get("theater", 0) + 1
    return [
        AreaStat(
            source="vtb_ood_v2",
            model=role,
            area=category,
            n=bucket["n"],
            errors=bucket["errors"],
            classes=bucket["classes"],
        )
        for (role, category), bucket in sorted(buckets.items())
    ]


def build_table(
    paths: Sequence[Path],
    *,
    min_n: int = DEFAULT_MIN_N,
    sources: Optional[Iterable[str]] = None,
) -> BedTable:
    """Build the bed from the given result files. Missing files are skipped, not faked."""
    areas: List[AreaStat] = []
    used: List[str] = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        if path.name == DEFAULT_PRINCIPLE_BENCH.name:
            areas.extend(load_principle_bench(path))
        elif path.name == DEFAULT_VTB_OOD_V2.name:
            areas.extend(load_vtb_ood_v2(path))
        else:
            label = path.stem.replace("_benchmark_results", "").replace("vtb_", "vtb_")
            areas.extend(load_vtb_case_table(path, source=label))
        used.append(path.name)
    if sources is not None:
        used = list(sources)
    return BedTable(
        min_n=min_n,
        areas=sorted(areas, key=lambda a: a.key),
        sources=used,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def default_paths() -> List[Path]:
    return [
        DEFAULT_PRINCIPLE_BENCH,
        DEFAULT_VTB_QWEN,
        DEFAULT_VTB_LFM,
        DEFAULT_VTB_OOD_V2,
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Statistical bed — error rates per area.")
    parser.add_argument("--build", action="store_true", help="write area_error_table.json/.md")
    parser.add_argument("--print", dest="do_print", action="store_true", help="print the markdown table")
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument("--source", action="append", default=None, help="explicit result file(s)")
    args = parser.parse_args(argv)

    paths = [Path(s) for s in args.source] if args.source else default_paths()
    table = build_table(paths, min_n=args.min_n)

    if not table.areas:
        print("No result files found — nothing measured, nothing to weigh.")
        return 1

    if args.do_print or not args.build:
        print(table.to_markdown())

    if args.build:
        OUT_JSON.write_text(
            json.dumps(table.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        OUT_MD.write_text(table.to_markdown() + "\n", encoding="utf-8")
        print(f"written: {OUT_JSON.name}, {OUT_MD.name}")
        print(f"areas measured={len(table.areas)} weighted={len(table.sufficient())}"
              f" insufficient={len(table.insufficient())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
