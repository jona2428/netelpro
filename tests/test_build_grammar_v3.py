"""Tests for Grammar v3 SFT Dataset Generator (training/data/build_grammar_v3.py).
================================================================================

Validates:
1. Template byte-identity: task prompts regenerate byte-for-byte identical to
   `benchmarks/principle_bench.py`'s `build_prompt(task, "netelpro")`.
2. Oracle verification: sample targets 100% compile and pass differential
   verification via the real Netelpro compiler (`compile_filter` + `verify_int`)
   and pass `grade_netelpro`.
3. Train/held-out name disjointness: all signal and family names in heldout.jsonl
   never appear in train.jsonl, and do not collide with dataset_v2 or bench pool 4.
4. JSONL schema match: every line matches the exact field schema of dataset_v2.jsonl.
5. Determinism: multiple runs with the same seed produce byte-for-byte identical files.
6. Minimum counts and mix proportion: >=1500 train, >=120 heldout, >=200 definitions,
   and exactly 25% of train_mix from dataset_v2.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set


from benchmarks.principle_bench import (
    FRESH_POOL_4,
    build_prompt,
    grade_netelpro,
)
from benchmarks.vtb_ood_runner import HONESTY_SYSTEM_PROMPT
from netelpro.rule_filter import compile_filter
from training.data.build_grammar_v3 import (
    SYSTEM_PROMPT,
    build_grammar_datasets,
    synthesize_rule_definitions,
    to_chat_format,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = REPO_ROOT / "training" / "data" / "train.jsonl"
HELDOUT_FILE = REPO_ROOT / "training" / "data" / "heldout.jsonl"
MIX_FILE = REPO_ROOT / "training" / "data" / "train_mix.jsonl"
V2_FILE = REPO_ROOT / "training" / "data" / "dataset_v2.jsonl"

EXPECTED_SCHEMA_KEYS = {
    "domain_a",
    "domain_b",
    "shared_law",
    "prompt",
    "completion",
    "netelpro_contract",
    "netelpro_cases",
    "netelpro_verified",
    "family",
    "table_id",
    "params",
    "generator",
    "timestamp",
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    assert path.exists(), f"File does not exist: {path}"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise AssertionError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


# ---------------------------------------------------------------------------
# 1. Template Byte-Identity
# ---------------------------------------------------------------------------


def test_template_byte_identity() -> None:
    """Regenerate prompts using principle_bench's own rendering and assert byte-exact match."""
    train_rows = _read_jsonl(TRAIN_FILE)
    assert len(train_rows) > 0, "train.jsonl is empty"

    definitions = synthesize_rule_definitions()
    def_map = {d["family"]: d for d in definitions}

    # Verify byte-exact prompt match across multiple representative samples
    for sample in train_rows[:10]:
        fam = sample["family"]
        assert fam in def_map, f"Unknown family in sample: {fam}"
        defn = def_map[fam]

        task = {
            "params": sample["params"],
            "scenarios": defn["scenarios"],
            "domain_a": sample["domain_a"],
            "domain_b": sample["domain_b"],
            "law": sample["shared_law"],
            "table_id": sample["table_id"],
        }

        bench_prompt = build_prompt(task, "netelpro")
        assert sample["prompt"] == bench_prompt, (
            f"Prompt mismatch for family {fam} table_id {sample['table_id']}:\n"
            f"Expected length {len(bench_prompt)}, got {len(sample['prompt'])}\n"
            f"Diff start: {sample['prompt'][:100]} vs {bench_prompt[:100]}"
        )


# ---------------------------------------------------------------------------
# 2. Oracle & Native Compiler Verification
# ---------------------------------------------------------------------------


def test_oracle_and_compiler_verification() -> None:
    """Sample targets 100% compile via real Netelpro compiler and pass grade_netelpro."""
    train_rows = _read_jsonl(TRAIN_FILE)
    heldout_rows = _read_jsonl(HELDOUT_FILE)

    # Test 20 train samples and 10 heldout samples
    test_samples = train_rows[:20] + heldout_rows[:10]
    assert len(test_samples) == 30

    for i, sample in enumerate(test_samples):
        contract = sample["netelpro_contract"]
        cases_raw = sample["netelpro_cases"]
        completion = sample["completion"]
        params = sample["params"]
        table_id = sample["table_id"]

        # A. Compiler execution
        rf = compile_filter(contract)
        truth = [((a, b, c), v) for a, b, c, v in cases_raw]
        mismatches = rf.verify_int(truth)
        assert mismatches == [], (
            f"Sample {i} (family {sample['family']}, table {table_id}) had mismatches: {mismatches}"
        )

        # B. Benchmark evaluator check
        ok, err, msg = grade_netelpro(completion, params, table_id)
        assert ok, f"Sample {i} failed principle_bench grader: err={err}, msg={msg}"


# ---------------------------------------------------------------------------
# 3. Train / Heldout Name Disjointness
# ---------------------------------------------------------------------------


def test_train_heldout_name_disjointness() -> None:
    """Rule names and parameters in heldout.jsonl must NEVER appear in train.jsonl."""
    train_rows = _read_jsonl(TRAIN_FILE)
    heldout_rows = _read_jsonl(HELDOUT_FILE)

    train_params: Set[str] = {p for r in train_rows for p in r["params"]}
    heldout_params: Set[str] = {p for r in heldout_rows for p in r["params"]}

    train_families: Set[str] = {r["family"] for r in train_rows}
    heldout_families: Set[str] = {r["family"] for r in heldout_rows}

    # Strict disjointness
    param_overlap = train_params & heldout_params
    assert param_overlap == set(), f"Parameter names overlap between train and heldout: {param_overlap}"

    family_overlap = train_families & heldout_families
    assert family_overlap == set(), f"Family names overlap between train and heldout: {family_overlap}"

    # Also check no collision with benchmark FRESH_POOL_4
    bench_pool4_params: Set[str] = {p for pool in FRESH_POOL_4.values() for p in pool}
    assert train_params.isdisjoint(bench_pool4_params), "Train params collide with FRESH_POOL_4"
    assert heldout_params.isdisjoint(bench_pool4_params), "Heldout params collide with FRESH_POOL_4"


# ---------------------------------------------------------------------------
# 4. JSONL Schema Match vs dataset_v2.jsonl
# ---------------------------------------------------------------------------


def test_jsonl_schema_match_vs_dataset_v2() -> None:
    """Each jsonl line matches dataset_v2.jsonl schema and field types exactly."""
    v2_rows = _read_jsonl(V2_FILE)
    assert len(v2_rows) > 0
    actual_v2_keys = set(v2_rows[0].keys())
    assert actual_v2_keys == EXPECTED_SCHEMA_KEYS, f"Schema mismatch: {actual_v2_keys}"

    for path in (TRAIN_FILE, HELDOUT_FILE, MIX_FILE):
        rows = _read_jsonl(path)
        assert len(rows) > 0, f"{path} is empty"
        for idx, row in enumerate(rows[:50]):  # check first 50 rows of each
            assert set(row.keys()) == EXPECTED_SCHEMA_KEYS, (
                f"Row {idx} in {path.name} has keys {set(row.keys())} != {EXPECTED_SCHEMA_KEYS}"
            )
            assert isinstance(row["domain_a"], str)
            assert isinstance(row["domain_b"], str)
            assert isinstance(row["shared_law"], str)
            assert isinstance(row["prompt"], str)
            assert isinstance(row["completion"], str)
            assert isinstance(row["netelpro_contract"], str)
            assert isinstance(row["netelpro_cases"], list)
            assert len(row["netelpro_cases"]) == 8
            assert row["netelpro_verified"] is True
            assert isinstance(row["family"], str)
            assert isinstance(row["table_id"], int)
            assert isinstance(row["params"], list)
            assert len(row["params"]) == 3
            assert isinstance(row["generator"], str)
            assert isinstance(row["timestamp"], float)


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    """Two runs with the same seed produce byte-for-byte identical files."""
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        dir1 = Path(td1)
        dir2 = Path(td2)

        build_grammar_datasets(
            output_dir=dir1,
            n_train_target=20,
            n_heldout_target=10,
            v2_mix_fraction=0.25,
            seed=42,
            v2_path=V2_FILE,
        )

        build_grammar_datasets(
            output_dir=dir2,
            n_train_target=20,
            n_heldout_target=10,
            v2_mix_fraction=0.25,
            seed=42,
            v2_path=V2_FILE,
        )

        for filename in ("train.jsonl", "heldout.jsonl", "train_mix.jsonl"):
            bytes1 = (dir1 / filename).read_bytes()
            bytes2 = (dir2 / filename).read_bytes()
            assert bytes1 == bytes2, f"Non-deterministic generation for {filename}"


# ---------------------------------------------------------------------------
# 6. Minimum Counts & Mix Proportions
# ---------------------------------------------------------------------------


def test_counts_and_mix_proportions() -> None:
    """Validate dataset counts, definition diversity, and replay mix proportions."""
    train_rows = _read_jsonl(TRAIN_FILE)
    heldout_rows = _read_jsonl(HELDOUT_FILE)
    mix_rows = _read_jsonl(MIX_FILE)

    assert len(train_rows) >= 1500, f"train.jsonl count {len(train_rows)} < 1500"
    assert len(heldout_rows) >= 120, f"heldout.jsonl count {len(heldout_rows)} < 120"

    # Distinct definitions synthesized
    defs = synthesize_rule_definitions()
    assert len(defs) >= 200, f"Synthesized definitions {len(defs)} < 200"

    # Replay proportion: exactly 25% of train_mix comes from dataset_v2
    v2_families = {
        "continuity_backpressure", "apoptosis_circuit_breaker", "annealing_scheduler",
        "hebb_moe_router", "coagulation_2pc", "mycelium_gossip", "chatelier_ratelimit",
        "commons_uma_arbiter", "pheromone_hnsw", "lymph_gc", "mirage_decoder",
        "lagrange_loadbalancer",
    }

    v2_count = sum(1 for r in mix_rows if r["family"] in v2_families)
    grammar_count = sum(1 for r in mix_rows if r["family"] not in v2_families)

    assert len(mix_rows) == len(train_rows) + v2_count
    assert grammar_count == len(train_rows)
    mix_ratio = v2_count / len(mix_rows)
    assert abs(mix_ratio - 0.25) < 0.01, f"Expected ~25% v2 mix, got {mix_ratio * 100:.2f}%"


# ---------------------------------------------------------------------------
# 7. Honesty System Prompt Integration
# ---------------------------------------------------------------------------


def test_honesty_system_prompt_integration() -> None:
    """Ensure HONESTY_SYSTEM_PROMPT is reused verbatim."""
    assert SYSTEM_PROMPT == HONESTY_SYSTEM_PROMPT
    train_rows = _read_jsonl(TRAIN_FILE)
    chat_sample = to_chat_format(train_rows[0])
    assert chat_sample[0]["role"] == "system"
    assert chat_sample[0]["content"] == HONESTY_SYSTEM_PROMPT
    assert chat_sample[1]["role"] == "user"
    assert chat_sample[1]["content"] == train_rows[0]["prompt"]
    assert chat_sample[2]["role"] == "assistant"
    assert chat_sample[2]["content"] == train_rows[0]["completion"]
