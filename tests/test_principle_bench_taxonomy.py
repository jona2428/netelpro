# tests/test_principle_bench_taxonomy.py
"""Contract tests for the error taxonomy of principle_bench.grade_netelpro.

Why these exist (measured 2026-09-17 over benchmarks/principle_bench_results.json,
the 2026-09-14 run artifact):

`"```netelpro" in response` is a substring test, and "```netelpro" is a literal
prefix of "```netelpro-cases". So a response whose CONTRACT fence is perfectly
closed, which merely names the cases fence in prose, fell through to the
`truncado` branch carrying the detail "fence cases sin cerrar" -- a message that
contradicts itself, because that branch is only reachable after the contract
fence matched `(.*?)``` `, i.e. after it closed.

The mislabel is not cosmetic: `truncado` was reported as the dominant failure
mode in 9 of the 12 families of the statistical bed that feeds Phase E, while
the eval's actual largest error class is `sin_casos` (106/360 in the diagnostic,
the number the run #5 exit gate is held to). Of the 137 `truncado` labels in the
artifact, 128 carried the self-contradicting detail. The taxonomy must not
report a missing cases block as truncation.

The opening fences are matched specifically (negative lookahead), never by
substring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.principle_bench import (  # noqa: E402
    FENCE_CASES_OPEN,
    FENCE_NP_OPEN,
    grade_netelpro,
)

TRAIN_JSONL = REPO_ROOT / "training" / "data" / "train.jsonl"


def _gold_sample():
    """One real gold sample from the dataset the model is trained on.

    Skips (never fails) when the dataset is absent: it is a training artifact,
    and a fresh clone must not depend on it to run the taxonomy contract.
    """
    if not TRAIN_JSONL.exists():
        pytest.skip("training/data/train.jsonl not present in this checkout")
    line = json.loads(TRAIN_JSONL.read_text(encoding="utf-8").splitlines()[0])
    return line["completion"], line["params"], line["table_id"]


# ---------------------------------------------------------------------------
# The opening-fence patterns are specific, not substring
# ---------------------------------------------------------------------------


def test_contract_fence_pattern_does_not_match_the_cases_fence():
    """The exact defect: ```netelpro-cases must NOT register as a contract fence."""
    assert FENCE_NP_OPEN.search("```netelpro\n(rule)\n```") is not None
    assert FENCE_NP_OPEN.search("```netelpro-cases\n(0,0,0) -> 0\n```") is None
    assert FENCE_CASES_OPEN.search("```netelpro-cases\n(0,0,0) -> 0\n```") is not None
    # A bare contract fence must not be mistaken for a cases fence either.
    assert FENCE_CASES_OPEN.search("```netelpro\n(rule)\n```") is None


def test_cases_fence_alone_is_sin_bloque_not_truncado():
    """Only the cases fence opened => the contract was never produced."""
    ok, err, _ = grade_netelpro("```netelpro-cases\n(0,0,0) -> 0\n```", ["a"], 1)
    assert ok is False
    assert err == "sin_bloque"


# ---------------------------------------------------------------------------
# The mislabel, on a real gold sample
# ---------------------------------------------------------------------------


def test_closed_contract_plus_prose_mention_is_sin_casos_not_truncado():
    """THE REGRESSION: contract closed, cases fence only named in prose.

    Nothing is unterminated here, so `truncado` is a false statement. The honest
    label is `sin_casos`: the contract is present, the cases block is not.
    """
    comp, params, table_id = _gold_sample()
    contract = comp.split("```netelpro-cases")[0].rstrip()
    response = contract + "\nA continuacion el bloque netelpro-cases:\n"

    ok, err, detail = grade_netelpro(response, params, table_id)
    assert ok is False
    assert err == "sin_casos", f"expected sin_casos, got {err!r} ({detail!r})"


def test_closed_contract_without_any_mention_is_sin_casos():
    comp, params, table_id = _gold_sample()
    contract = comp.split("```netelpro-cases")[0].rstrip()

    ok, err, _ = grade_netelpro(contract + "\nListo.\n", params, table_id)
    assert ok is False
    assert err == "sin_casos"


# ---------------------------------------------------------------------------
# Real truncation is still truncation (the fix must not swallow it)
# ---------------------------------------------------------------------------


def test_cases_fence_left_open_is_still_truncado():
    """A genuinely unterminated cases fence keeps the truncation label."""
    comp, params, table_id = _gold_sample()
    contract = comp.split("```netelpro-cases")[0].rstrip()
    cases_block = "```netelpro-cases" + comp.split("```netelpro-cases")[1]
    cut = contract + "\n" + cases_block.rstrip().rstrip("`")

    ok, err, detail = grade_netelpro(cut, params, table_id)
    assert ok is False
    assert err == "truncado"
    assert detail == "fence cases sin cerrar"


def test_contract_fence_left_open_is_still_truncado():
    comp, params, table_id = _gold_sample()
    cut = comp.split("```netelpro-cases")[0].rstrip().rstrip("`")

    ok, err, detail = grade_netelpro(cut, params, table_id)
    assert ok is False
    assert err == "truncado"
    assert detail == "fence netelpro sin cerrar"


# ---------------------------------------------------------------------------
# The gold sample must still pass: the fix cannot change a valid verdict
# ---------------------------------------------------------------------------


def test_gold_sample_still_verifies_against_the_real_compiler():
    """Full gold generation: contract + cases, both fences closed."""
    comp, params, table_id = _gold_sample()

    ok, err, detail = grade_netelpro(comp, params, table_id)
    assert ok is True, f"gold sample rejected: err={err!r} detail={detail!r}"
    assert err == "ok"


def test_no_fence_at_all_is_sin_bloque():
    ok, err, _ = grade_netelpro("No puedo formalizar esta regla.", ["a"], 1)
    assert ok is False
    assert err == "sin_bloque"